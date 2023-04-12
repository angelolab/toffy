# adapted from https://machinelearningmastery.com/curve-fitting-with-python/
import copy
import os
import warnings

import matplotlib.pyplot as plt
import natsort as ns
import numpy as np
import pandas as pd
from alpineer import image_utils, io_utils, load_utils, misc_utils
from mibi_bin_tools.bin_files import extract_bin_files, get_median_pulse_height
from mibi_bin_tools.panel_utils import make_panel
from scipy.optimize import curve_fit

from toffy.json_utils import check_for_empty_files, read_json_file, write_json_file


def write_counts_per_mass(base_dir, output_dir, fov, masses, start_offset=0.5, stop_offset=0.5):
    """Records the total counts per mass for the specified FOV

    Args:
        base_dir (str): the directory containing the FOV
        output_dir (str): the directory where the csv file will be saved
        fov (str): the name of the fov to extract
        masses (list): the list of masses to extract counts from
        start_offset (float): beginning value for integrating mass peak
        stop_offset (float): ending value for integrating mass peak
    """

    # create panel with extraction criteria
    panel = make_panel(mass=masses, low_range=start_offset, high_range=stop_offset)

    array = extract_bin_files(
        data_dir=base_dir,
        out_dir=None,
        include_fovs=[fov],
        panel=panel,
        intensities=False,
    )
    # we only care about pulse counts, not intensities
    array = array.loc[fov, "pulse", :, :, :]
    channel_count = np.sum(array, axis=(0, 1))

    # create df to hold output
    fovs = np.repeat(fov, len(masses))
    out_df = pd.DataFrame({"mass": masses, "fov": fovs, "channel_count": channel_count})
    out_df.to_csv(os.path.join(output_dir, fov + "_channel_counts.csv"), index=False)


def write_mph_per_mass(
    base_dir, output_dir, fov, masses, start_offset=0.5, stop_offset=0.5, proficient=False
):
    """Records the median pulse height (MPH) per mass for the specified FOV

    Args:
        base_dir (str): the directory containing the FOV
        output_dir (str): the directory where the csv file will be saved
        fov (str): the name of the fov to extract
        masses (list): the list of masses to extract MPH from
        start_offset (float): beginning value for calculating mph values
        stop_offset (float): ending value for calculating mph values
        proficient (bool): whether proficient MPH data is written or not
    """
    # hold computed values
    mph_vals = []

    # compute pulse heights
    panel = make_panel(
        mass=masses, target_name=masses, low_range=start_offset, high_range=stop_offset
    )
    for mass in masses:
        mph_vals.append(
            get_median_pulse_height(data_dir=base_dir, fov=fov, channel=mass, panel=panel)
        )
    # create df to hold output
    fovs = np.repeat(fov, len(masses))
    out_df = pd.DataFrame({"mass": masses, "fov": fovs, "pulse_height": mph_vals})
    pulse_heights_file = (
        fov + "_pulse_heights_proficient.csv" if proficient else fov + "_pulse_heights.csv"
    )
    out_df.to_csv(os.path.join(output_dir, pulse_heights_file), index=False)


def create_objective_function(obj_func):
    """Creates a function of specified type to be used for fitting a curve

    Args:
        obj_func (str): the desired objective function. Must be either poly_2, ..., poly_5, or log

    Returns:
        function: the function which will be optimized"""

    # input validation
    valid_funcs = ["poly_2", "poly_3", "poly_4", "poly_5", "log", "exp"]
    if obj_func not in valid_funcs:
        raise ValueError("Invalid function, must be one of {}".format(valid_funcs))

    # define objective functions
    def poly_2(x, a, b, c):
        return a * x + b * x**2 + c

    def poly_3(x, a, b, c, d):
        return a * x + b * x**2 + c * x**3 + d

    def poly_4(x, a, b, c, d, e):
        return a * x + b * x**2 + c * x**3 + d * x**4 + e

    def poly_5(x, a, b, c, d, e, f):
        return a * x + b * x**2 + c * x**3 + d * x**4 + e * x**5 + f

    def log(x, a, b):
        # edge case appears when x is a single int/float that returns non NaN/inf for np.log
        # in this case, return the base np.log calculation
        try:
            return (a * np.ma.log(x) + b).filled(0)
        except AttributeError:
            return a * np.log(x) + b

    def exp(x, a, b, c, d):
        try:
            x_log = (np.ma.log(x)).filled(0)
        except AttributeError:
            x_log = a * np.log(x) + b

        return a * x_log + b * x_log**2 + c * x_log**3 + d

    objectives = {
        "poly_2": poly_2,
        "poly_3": poly_3,
        "poly_4": poly_4,
        "poly_5": poly_5,
        "log": log,
        "exp": exp,
    }

    return objectives[obj_func]


def fit_calibration_curve(
    x_vals,
    y_vals,
    obj_func,
    outliers=None,
    plot_fit=False,
    save_path=None,
    x_label=None,
    y_label=None,
    title=None,
    show_plot=False,
):
    """Finds the optimal weights to fit the supplied values for the specified function

    Args:
        x_vals (list): the x values to be fit
        y_vals (list): the y value to be fit
        obj_func (str): the name of the function that will be fit to the data
        outliers (tuple or None): optional tuple of ([x_coords], [y_coords]) to plot
        plot_fit (bool): whether or not to plot the fit of the function vs the values
        save_path (str or None): location to save the plot of the fitted values
        x_label (str or None): label for the x-axis
        y_label (str or None):label for the y-axis
        title (str or None): label for the plot title
        show_plot (bool): whether to show plot, default False

    Returns:
        list: the weights of the fitted function"""

    # get objective function
    objective = create_objective_function(obj_func)

    # get fitted values
    popt, _ = curve_fit(objective, x_vals, y_vals)

    # plot overlay of predicted fit and real values
    if plot_fit:
        plt.scatter(x_vals, y_vals)
        x_line = np.arange(min(x_vals), max(x_vals), 1)
        y_line = objective(x_line, *popt)
        plt.plot(x_line, y_line, "--", color="red")

        # add labels
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        if title:
            plt.title(title)

        if outliers is not None:
            plt.scatter(outliers[0], outliers[1])

        if save_path is not None:
            plt.savefig(save_path)

        if show_plot:
            plt.show()
        plt.close()

    return popt


def create_prediction_function(name, weights):
    """Creates a prediction function given a specified function type and fitted weights

    Args:
        name (str): name of the function to use
        weights (list): list of fitted weights

    Returns:
        func: prediction function"""

    # get function based on specified pred type
    obj_func = create_objective_function(name)

    # define function which takes only x as input, uses weights for remaining parameters
    def pred_func(x):
        output = obj_func(x, *weights)
        return output

    return pred_func


def combine_run_metrics(run_dir, substring):
    """Combines the specified metrics from different FOVs into a single file

    Args:
        run_dir (str): the directory containing the files
        substring(str): the substring contained within the files to be combined"""

    files = io_utils.list_files(run_dir, substring)

    # validate inputs
    if len(files) == 0:
        raise ValueError("No files found in {}".format(run_dir))

    if substring + "_combined.csv" in files:
        warnings.warn(
            "Removing previously generated combined {} file in {}".format(substring, run_dir)
        )
        os.remove(os.path.join(run_dir, substring + "_combined.csv"))
        files = [file for file in files if "combined" not in file]

    # collect all metrics files
    metrics = []
    for file in files:
        metrics.append(pd.read_csv(os.path.join(run_dir, file)))

    # check that all files are the same length
    if len(metrics) > 1:
        base_len = len(metrics[0])
        for i in range(1, len(metrics)):
            if len(metrics[i]) != base_len:
                raise ValueError(
                    "Not all {} files are the same length: file {} does not matchfile {}".format(
                        substring, files[0], files[i]
                    )
                )

    metrics = pd.concat(metrics)

    metrics.to_csv(os.path.join(run_dir, substring + "_combined.csv"), index=False)


def combine_tuning_curve_metrics(dir_list, count_range):
    """Combines metrics together into a single dataframe for fitting a turning curve

    Args:
        dir_list (list): list of directories to pull metrics from
        count_range (tuple): range of appropriate mass count values for the FOVs

    Returns:
        pd.DataFrame: dataframe containing aggregates metrics"""

    # create list to hold all extracted data
    all_dirs, excluded_fovs = [], []
    # loop through each run folder
    for dir in dir_list:
        pulse_heights = pd.read_csv(os.path.join(dir, "fov-1-scan-1_pulse_heights.csv"))
        channel_counts = pd.read_csv(os.path.join(dir, "fov-1-scan-1_channel_counts.csv"))

        # check for extreme count values
        extreme_val = False
        if count_range:
            counts = channel_counts.channel_count
            if any(counts <= count_range[0]) or any(counts >= count_range[1]):
                extreme_val = True
                excluded_fovs.append(os.path.basename(dir))
        # do not add extreme value fovs to the combined data frame
        if extreme_val:
            continue

        # combine tables together
        combined = pulse_heights.merge(channel_counts, "outer", on=["fov", "mass"])
        if len(combined) != len(pulse_heights):
            raise ValueError(
                "Pulse heights and channel counts must be generated for the same "
                "mass ranges and fovs. However, the data the following does do not "
                "exactly match: {}".format(dir)
            )

        # add directory label and add to list
        combined["directory"] = dir
        all_dirs.append(combined)

    if len(excluded_fovs) > 0:
        excluded_fovs = ns.natsorted(excluded_fovs)
        print(
            (
                "The counts for the FOV contained in the following folders are outside of "
                "the expected range and will be excluded: "
            ),
            *excluded_fovs,
            sep="\n- ",
        )
    elif count_range:
        print("No extreme values detected.")

    # combine data from each dir together
    all_data = pd.concat(all_dirs)
    all_data.reset_index()

    # check for sufficient data
    if len(set(all_data.directory)) < 4:
        raise ValueError("Invalid amount of FOV data. Please choose another sweep.")

    # create normalized counts column
    subset = all_data[["channel_count", "mass"]]
    all_data["norm_channel_count"] = subset.groupby("mass").transform(lambda x: (x / x.max()))

    return all_data


def plot_voltage_vs_counts(sweep_fov_paths, combined_data, save_path):
    """Creates a barplot of voltage and maximum channel counts and saves the image

    Args:
        sweep_fov_paths (list): paths to the sweep folders
        combined_data (pd.DataFrame): combined data frame of pulse height and channel info
        save_path (str): where to save plot to
    """

    voltages, max_channel_counts = [], []

    # get voltage and max channel count for each fov
    for fov_path in sweep_fov_paths:
        fov_data = read_json_file(os.path.join(fov_path, "fov-1-scan-1.json"))

        # locate index storing the detector voltage
        for j in range(0, len(fov_data["hvDac"])):
            if fov_data["hvDac"][j]["name"] == "Detector":
                index = j
                break
        fov_voltage = fov_data["hvDac"][index]["currentSetPoint"]
        fov_counts = combined_data.loc[combined_data["directory"] == fov_path, "channel_count"]

        voltages.append(fov_voltage)
        max_channel_counts.append(fov_counts.max() / 1000000)

    plt.bar(voltages, max_channel_counts)
    plt.xlabel("Voltage")
    plt.ylabel("Maximum Channel Counts (in millions)")
    plt.savefig(save_path)
    plt.close()


def show_multiple_plots(rows, cols, image_paths, image_size=(17, 12)):
    """Displays a single image with multiple plots

    Args:
        rows (int): number of rows of plot grid
        cols (int): number of columns of plot grid
        image_paths (list): list of paths to the previously saved plot images
        image_size (tuple): length and width of the new image produced
    """

    fig = plt.figure(figsize=image_size)

    # add each plot to the figure
    for img, path in enumerate(image_paths):
        plot = plt.imread(path)
        fig.add_subplot(rows, cols, img + 1)
        plt.imshow(plot)
        plt.axis("off")

    plt.show()


def create_tuning_function(
    sweep_path,
    moly_masses=[92, 94, 95, 96, 97, 98, 100],
    save_path=os.path.join("..", "toffy", "norm_func.json"),
    count_range=(0, 3000000),
):
    """Creates a tuning curve for an instrument based on the provided moly sweep

    Args:
        sweep_path (str): path to folder containing a detector sweep
        moly_masses (list): list of masses to use for fitting the curve
        save_path (str): path to save the weights of the tuning curve
        count_range (tuple): range of appropriate mass count values for the FOVs
    """

    # get all folders from the sweep
    sweep_fovs = io_utils.list_folders(sweep_path)
    sweep_fov_paths = [os.path.join(sweep_path, fov) for fov in sweep_fovs]
    # check for sufficient directory structure
    if len(sweep_fovs) < 4:
        raise ValueError("Invalid amount of FOV folders. Please use at least 4 voltages.")

    # compute pulse heights and channel counts for each FOV if files don't already exist
    for fov_path in sweep_fov_paths:
        if not os.path.exists(
            os.path.join(fov_path, "fov-1-scan-1_pulse_heights.csv")
        ) or not os.path.exists(os.path.join(fov_path, "fov-1-scan-1_channel_counts.csv")):
            # check for bin file in each folder
            bin_file = io_utils.list_files(fov_path, substrs="bin")
            if len(bin_file) == 0:
                raise ValueError(f"No bin file detected in {fov_path}")

            write_mph_per_mass(
                base_dir=fov_path,
                output_dir=fov_path,
                fov="fov-1-scan-1",
                masses=moly_masses,
            )
            write_counts_per_mass(
                base_dir=fov_path,
                output_dir=fov_path,
                fov="fov-1-scan-1",
                masses=moly_masses,
            )

    if count_range:
        # combine all data together into single df for comparison
        all_data = combine_tuning_curve_metrics(sweep_fov_paths, count_range=None)

        # generate curve with extreme values included
        all_coeffs = fit_calibration_curve(
            all_data["pulse_height"].values,
            all_data["norm_channel_count"].values,
            "exp",
            plot_fit=True,
            save_path=os.path.join(sweep_path, "function_fit_all_data.jpg"),
            x_label="Median Pulse Height",
            y_label="Normalized Channel Counts",
            title="Tuning Curve (all data)",
            show_plot=False,
        )

        # plot voltage against maximum channel count
        plot_voltage_vs_counts(
            sweep_fov_paths,
            all_data,
            save_path=os.path.join(sweep_path, "voltage_vs_counts.jpg"),
        )

        show_multiple_plots(
            1,
            2,
            [
                os.path.join(sweep_path, "function_fit_all_data.jpg"),
                os.path.join(sweep_path, "voltage_vs_counts.jpg"),
            ],
        )

    # combine tuning date into single df, if count_range given then extreme values excluded
    tuning_data = combine_tuning_curve_metrics(sweep_fov_paths, count_range=count_range)

    # generate fitted curve
    tuning_coeffs = fit_calibration_curve(
        tuning_data["pulse_height"].values,
        tuning_data["norm_channel_count"].values,
        "exp",
        plot_fit=True,
        save_path=os.path.join(sweep_path, "function_fit.jpg"),
        x_label="Median Pulse Height",
        y_label="Normalized Channel Counts",
        title="Tuning Curve",
        show_plot=True,
    )

    # save the fitted curve
    norm_json = {"name": "exp", "weights": tuning_coeffs.tolist()}
    write_json_file(json_path=save_path, json_object=norm_json)


def identify_outliers(x_vals, y_vals, obj_func, outlier_fraction=0.1):
    """Finds the indices of outliers in the provided data to prune for subsequent curve fitting

    Args:
        x_vals (np.array): the x values of the data being analyzed
        y_vals (np.array): the y values of the data being analyzed
        obj_func (str): the objective function to use for curve fitting to determine outliers
        outlier_fraction (float): the fractional deviation from predicted value required in
            order to classify a data point as an outlier

    Returns:
        np.array: the indices of the identified outliers"""

    # get objective function
    objective = create_objective_function(obj_func)

    # get fitted values
    popt, _ = curve_fit(objective, x_vals, y_vals)

    # create generate function
    func = create_prediction_function(name=obj_func, weights=popt)

    # generate predictions
    preds = func(x_vals)

    # specify outlier bounds based on multiple of predicted value
    upper_bound = preds * (1 + outlier_fraction)
    lower_bound = preds * (1 - outlier_fraction)

    # identify outliers
    outlier_mask = np.logical_or(y_vals > upper_bound, y_vals < lower_bound)
    outlier_idx = np.where(outlier_mask)[0]

    return outlier_idx


def smooth_outliers(vals, outlier_idx, smooth_range=2):
    """Performs local smoothing on the provided outliers

    Args:
        vals (np.array): the complete list of values to be smoothed
        outlier_idx (np.array): the indices of the outliers in *vals* argument
        smooth_range (int): the number of adjacent values in each direction to use for smoothing

    Returns:
        np.array: the smoothed version of the provided vals"""

    smoothed_vals = copy.deepcopy(vals)
    vals = np.array(vals)

    for outlier in outlier_idx:
        previous_vals = smoothed_vals[(outlier - smooth_range) : outlier]

        if outlier == len(vals):
            # last value in list, can't average using subsequent values
            subsequent_vals = []
        else:
            # not the last value, we can use remaining values to get an estimate
            subsequent_indices = np.arange(outlier + 1, len(vals))
            valid_subs_indices = [idx for idx in subsequent_indices if idx not in outlier_idx]
            subsequent_indices = np.array(valid_subs_indices)[:smooth_range]

            # check to make sure there are valid subsequent indices
            if len(subsequent_indices) > 0:
                subsequent_vals = vals[subsequent_indices]
            else:
                subsequent_vals = np.array([])

        new_val = np.mean(np.concatenate([previous_vals, subsequent_vals]))
        smoothed_vals[outlier] = new_val

    return smoothed_vals


def fit_mass_mph_curve(mph_vals, mass, save_dir, obj_func, min_obs=10):
    """Fits a curve for the MPH over time for the specified mass

    Args:
        mph_vals (list): mph for each FOV in the run
        mass (str or int): the mass being fit
        save_dir (str): the directory to save the fit parameters
        obj_func (str): the function to use for constructing the fit
        min_obs (int): the minimum number of observations to fit a curve, otherwise uses median
    """

    fov_order = np.linspace(0, len(mph_vals) - 1, len(mph_vals))
    save_path = os.path.join(save_dir, str(mass) + "_mph_fit.jpg")

    if len(mph_vals) > min_obs:
        # find outliers in the MPH vals
        outlier_idx = identify_outliers(x_vals=fov_order, y_vals=mph_vals, obj_func=obj_func)

        # replace with local smoothing around that point
        smoothed_vals = smooth_outliers(vals=mph_vals, outlier_idx=outlier_idx)

        # if outliers identified, generate tuple to pass to plotting function
        if len(outlier_idx) > 0:
            outlier_x = fov_order[outlier_idx]
            outlier_y = mph_vals[outlier_idx]
            outlier_tup = (outlier_x, outlier_y)
        else:
            outlier_tup = None

        # fit curve
        weights = fit_calibration_curve(
            x_vals=fov_order,
            y_vals=smoothed_vals,
            obj_func=obj_func,
            outliers=outlier_tup,
            plot_fit=True,
            save_path=save_path,
        )

    else:
        # default to using the median instead for short runs with small number of FOVs
        mph_median = np.median(mph_vals)
        if obj_func == "poly_2":
            weight_len = 3
        elif obj_func == "poly_3":
            weight_len = 4
        else:
            raise ValueError("Unsupported objective function provided: {}".format(obj_func))

        # plot median
        plt.axhline(y=mph_median, color="r", linestyle="-")
        plt.plot(fov_order, mph_vals, ".")
        plt.savefig(save_path)
        plt.close()

        # all coefficients except intercept are 0
        weights = np.zeros(weight_len)
        weights[-1] = mph_median

    mass_json = {"name": obj_func, "weights": weights.tolist()}
    mass_path = os.path.join(save_dir, str(mass) + "_norm_func.json")

    write_json_file(json_path=mass_path, json_object=mass_json)


def create_fitted_mass_mph_vals(pulse_height_df, obj_func_dir):
    """Uses the mph curves for each mass to generate a smoothed mph estimate

    Args:
        pulse_height_df (pd.DataFrame): contains the MPH value per mass for all FOVs
        obj_func_dir (str): directory containing the curves generated for each mass

    Returns:
        pd.DataFrame: updated dataframe with fitted version of each MPH value for each mass
    """

    # get all masses
    masses = np.unique(pulse_height_df["mass"].values)

    # create column to hold fitted values
    pulse_height_df["pulse_height_fit"] = 0

    # create x axis values
    num_fovs = len(np.unique(pulse_height_df["fov"]))
    fov_order = np.linspace(0, num_fovs - 1, num_fovs)

    for mass in masses:
        # if channel-specific prediction function does not exist, set to 0
        mass_path = os.path.join(obj_func_dir, str(mass) + "_norm_func.json")
        mass_idx = pulse_height_df["mass"] == mass

        if not os.path.exists(mass_path):
            pulse_height_df.loc[mass_idx, "pulse_height_fit"] = 0.0
            continue

        # load channel-specific prediction function
        mass_json = read_json_file(mass_path)

        # compute predicted MPH
        name, weights = mass_json["name"], mass_json["weights"]
        pred_func = create_prediction_function(name=name, weights=weights)
        pred_vals = pred_func(fov_order)

        # update df
        pulse_height_df.loc[mass_idx, "pulse_height_fit"] = pred_vals

    return pulse_height_df


def create_fitted_pulse_heights_file(pulse_height_dir, panel_info, norm_dir, mass_obj_func):
    """Create a single file containing the pulse heights after fitting a curve per mass

    Args:
        pulse_height_dir (str): path to directory containing pulse height csvs
        panel_info (pd.DataFrame): the panel for this dataset
        norm_dir (str): the directory where normalized images will be saved
        mass_obj_func (str): the objective function used to fit the MPH over time per mass

    Returns:
        pd.DataFrame: the combined pulse heights file"""

    # create variables for mass fitting
    masses = panel_info["Mass"].values
    fit_dir = os.path.join(norm_dir, "curve_fits")
    os.makedirs(fit_dir)

    # combine fov-level files together
    combine_run_metrics(run_dir=pulse_height_dir, substring="pulse_heights")
    pulse_height_df = pd.read_csv(os.path.join(pulse_height_dir, "pulse_heights_combined.csv"))

    # order by FOV
    ordering = ns.natsorted((pulse_height_df["fov"].unique()))
    pulse_height_df["fov"] = pd.Categorical(
        pulse_height_df["fov"], ordered=True, categories=ordering
    )
    pulse_height_df = pulse_height_df.sort_values("fov")

    # loop over each mass, and fit a curve for MPH over the course of the run
    for mass in masses:
        mph_vals = pulse_height_df.loc[pulse_height_df["mass"] == mass, "pulse_height"].values

        # only create a _norm_func file if the mph_vals are non-zero
        if np.all(mph_vals == 0):
            warnings.warn("Skipping normalization for mass %s with all zero pulse heights" % mass)
            continue

        fit_mass_mph_curve(mph_vals=mph_vals, mass=mass, save_dir=fit_dir, obj_func=mass_obj_func)

    # update pulse_height_df to include fitted mph values
    pulse_height_df = create_fitted_mass_mph_vals(
        pulse_height_df=pulse_height_df, obj_func_dir=fit_dir
    )

    return pulse_height_df


def normalize_fov(img_data, norm_vals, norm_dir, fov, channels, extreme_vals):
    """Normalize a single FOV with provided normalization constants for each channel"""

    # create directory to hold normalized images
    output_fov_dir = os.path.join(norm_dir, fov)
    if os.path.exists(output_fov_dir):
        print("output directory {} already exists, data will be overwritten".format(output_fov_dir))
    else:
        os.makedirs(output_fov_dir)

    # check if any values are outside expected range
    extreme_mask = np.logical_or(norm_vals < extreme_vals[0], norm_vals > extreme_vals[1])
    if np.any(extreme_mask):
        bad_channels = np.array(channels)[extreme_mask]
        warnings.warn(
            "The following channel(s) had an extreme normalization "
            "value for fov {}. Manual inspection for accuracy is "
            "recommended: {}".format(fov, bad_channels)
        )

    # correct images and save, ensure that no division by zero happens
    norm_vals = norm_vals.astype(img_data.dtype)
    norm_div = norm_vals.reshape((1, 1, 1, len(norm_vals)))
    norm_vals_masked = np.ma.masked_values(x=norm_vals, value=0)
    normalized_images = np.ma.divide(img_data.values, norm_div).filled(0)

    for idx, chan in enumerate(channels):
        fname = os.path.join(output_fov_dir, chan + ".tiff")
        image_utils.save_image(fname, normalized_images[0, :, :, idx])

    # save logs
    log_df = pd.DataFrame({"channels": channels, "norm_vals": norm_vals})
    log_df.to_csv(os.path.join(output_fov_dir, "normalization_coefs.csv"), index=False)


def normalize_image_data(
    img_dir,
    norm_dir,
    pulse_height_dir,
    panel_info,
    img_sub_folder="",
    mass_obj_func="poly_2",
    extreme_vals=(0.4, 1.1),
    norm_func_path=os.path.join("..", "toffy", "norm_func.json"),
):
    """Normalizes image data based on median pulse height from the run and a tuning curve

    Args:
        img_dir (str): directory with the image data
        norm_dir (str): directory where the normalized images will be saved
        pulse_height_dir (str): directory containing per-fov pulse heights
        panel_info (pd.DataFrame): mapping between channels and masses
        mass_obj_func (str): class of function to use for modeling MPH over time per mass
        extreme_vals (tuple): determines the range for norm vals which will raise a warning
        norm_func_path (str): file containing the saved weights for the normalization function
    """

    # error checks
    if not os.path.exists(norm_func_path):
        raise ValueError(
            "No normalization function found. You will need to run "
            "section 3 of the 1_set_up_toffy.ipynb notebook to generate the "
            "necessary function before you can normalize your data"
        )

    # create normalization function for mapping MPH to counts
    norm_json = read_json_file(norm_func_path)

    img_fovs = io_utils.list_folders(img_dir, "fov")

    norm_weights, norm_name = norm_json["weights"], norm_json["name"]
    norm_func = create_prediction_function(norm_name, norm_weights)

    # combine pulse heights together into single df
    pulse_height_df = create_fitted_pulse_heights_file(
        pulse_height_dir=pulse_height_dir,
        panel_info=panel_info,
        norm_dir=norm_dir,
        mass_obj_func=mass_obj_func,
    )
    # add channel name to pulse_height_df
    renamed_panel = panel_info.rename({"Mass": "mass"}, axis=1)
    pulse_height_df = pulse_height_df.merge(renamed_panel, how="left", on=["mass"])
    pulse_height_df = pulse_height_df.sort_values("Target")

    # make sure FOVs used to construct tuning curve are same ones being normalized
    pulse_fovs = np.unique(pulse_height_df["fov"])
    misc_utils.verify_same_elements(image_data_fovs=img_fovs, pulse_height_csv_files=pulse_fovs)

    # loop over each fov
    for fov in img_fovs:
        # compute per-mass normalization constant
        pulse_height_fov = pulse_height_df.loc[pulse_height_df["fov"] == fov, :]
        channels = pulse_height_fov["Target"].values
        norm_vals = norm_func(pulse_height_fov["pulse_height_fit"].values)

        # get images
        images = load_utils.load_imgs_from_tree(
            img_dir, fovs=[fov], channels=channels, img_sub_folder=img_sub_folder
        )

        # normalize and save
        normalize_fov(
            img_data=images,
            norm_vals=norm_vals,
            norm_dir=norm_dir,
            fov=fov,
            channels=channels,
            extreme_vals=extreme_vals,
        )


def check_detector_voltage(run_dir):
    """Check all FOVs in a run to determine whether the detector voltage stays constant
    Args:
        run_dir(string): path to directory containing json files of all fovs in the run
    Return:
        raise error if changes in voltage were found between fovs
    """

    fovs = io_utils.remove_file_extensions(io_utils.list_files(run_dir, substrs=".bin"))
    changes_in_voltage = []

    # skip any damaged fovs
    empty_fovs = check_for_empty_files(run_dir)
    fovs = list(set(fovs).difference(empty_fovs))
    fovs = ns.natsorted(fovs)

    # check for voltage changes and add to list of dictionaries
    for i, fov in enumerate(fovs):
        fov_data = read_json_file(os.path.join(run_dir, fov + ".json"))

        # locate index storing the detector voltage
        for j in range(0, len(fov_data["hvDac"])):
            if fov_data["hvDac"][j]["name"] == "Detector":
                index = j
                break
        fov_voltage = fov_data["hvDac"][index]["currentSetPoint"]

        if i == 0:
            voltage_level = fov_voltage

        # detector voltage for current fov is different than previous
        elif fov_voltage != voltage_level:
            changes_in_voltage.append({fovs[i - 1]: voltage_level, fovs[i]: fov_voltage})
            voltage_level = fov_voltage

    err_str = ""
    for i, change in enumerate(changes_in_voltage):
        keys = list(change.keys())
        err_i = "Between {0} and {1} the voltage changed from {2} to {3}.".format(
            keys[0], keys[1], change[keys[0]], change[keys[1]]
        )
        err_str = err_str + "\n" + err_i

    # non-empty list of changes will raise an error
    if changes_in_voltage:
        raise ValueError("Changes in detector voltage were found during the run:\n" + err_str)
