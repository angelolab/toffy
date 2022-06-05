# adapted from https://machinelearningmastery.com/curve-fitting-with-python/

import json
import os
import shutil
import warnings

import numpy as np
from scipy.optimize import curve_fit
import skimage.io as io
import matplotlib.pyplot as plt
import natsort as ns
import pandas as pd

from ark.utils import io_utils, load_utils, misc_utils
from mibi_bin_tools.bin_files import extract_bin_files, get_median_pulse_height
from mibi_bin_tools.panel_utils import make_panel


def write_counts_per_mass(base_dir, output_dir, fov, masses, integration_window=(0.5, 0.5)):
    """Records the total counts per mass for the specified FOV

    Args:
        base_dir (str): the directory containing the FOV
        output_dir (str): the directory where the csv file will be saved
        fov (str): the name of the fov to extract
        masses (list): the list of masses to extract counts from
        integration_window (tuple): start and stop offset for integrating mass peak
    """

    start_offset, stop_offset = integration_window

    # create panel with extraction criteria
    panel = make_panel(mass=masses, low_range=start_offset, high_range=stop_offset)

    array = extract_bin_files(data_dir=base_dir, out_dir=None, include_fovs=[fov], panel=panel,
                              intensities=False)
    # we only care about pulse counts, not intensities
    array = array.loc[fov, 'pulse', :, :, :]
    channel_count = np.sum(array, axis=(0, 1))

    # create df to hold output
    fovs = np.repeat(fov, len(masses))
    out_df = pd.DataFrame({'mass': masses,
                           'fov': fovs,
                           'channel_count': channel_count})
    out_df.to_csv(os.path.join(output_dir, fov + '_channel_counts.csv'), index=False)


def write_mph_per_mass(base_dir, output_dir, fov, masses, integration_window=(0.5, 0.5)):
    """Records the mean pulse height (MPH) per mass for the specified FOV

    Args:
        base_dir (str): the directory containing the FOV
        output_dir (str): the directory where the csv file will be saved
        fov (str): the name of the fov to extract
        masses (list): the list of masses to extract MPH from
        integration_window (tuple): start and stop offset for integrating mass peak
    """
    start_offset, stop_offset = integration_window

    # hold computed values
    mph_vals = []

    for mass in masses:
        panel = make_panel(mass=mass, low_range=start_offset, high_range=stop_offset)
        mph_vals.append(get_median_pulse_height(data_dir=base_dir, fov=fov, channel='targ0',
                                                panel=panel))
    # create df to hold output
    fovs = np.repeat(fov, len(masses))
    out_df = pd.DataFrame({'mass': masses,
                           'fov': fovs,
                           'pulse_height': mph_vals})
    out_df.to_csv(os.path.join(output_dir, fov + '_pulse_heights.csv'), index=False)


def create_objective_function(obj_func):
    """Creates a function of specified type to be used for fitting a curve

    Args:
        obj_func (str): the desired objective function. Must be either poly_2, ..., poly_5, or log

    Returns:
        function: the function which will be optimized"""

    # input validation
    valid_funcs = ['poly_2', 'poly_3', 'poly_4', 'poly_5', 'log', 'exp']
    if obj_func not in valid_funcs:
        raise ValueError('Invalid function, must be one of {}'.format(valid_funcs))

    # define objective functions
    def poly_2(x, a, b, c):
        return a * x + b * x ** 2 + c

    def poly_3(x, a, b, c, d):
        return a * x + b * x ** 2 + c * x ** 3 + d

    def poly_4(x, a, b, c, d, e):
        return a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e

    def poly_5(x, a, b, c, d, e, f):
        return a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e * x ** 5 + f

    def log(x, a, b):
        return a * np.log(x) + b

    def exp(x, a, b, c, d):
        x_log = np.log(x)
        return a * x_log + b * x_log ** 2 + c * x_log ** 3 + d

    objectives = {'poly_2': poly_2, 'poly_3': poly_3, 'poly_4': poly_4, 'poly_5': poly_5,
                  'log': log, 'exp': exp}

    return objectives[obj_func]


def fit_calibration_curve(x_vals, y_vals, obj_func, plot_fit=False, save_path=None):
    """Finds the optimal weights to fit the supplied values for the specified function

    Args:
        x_vals (list): the x values to be fit
        y_vals (list): the y value to be fit
        obj_func (str): the name of the function that will be fit to the data
        plot_fit (bool): whether or not to plot the fit of the function vs the values
        save_path (str or None): location to save the plot of the fitted values

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
        plt.plot(x_line, y_line, '--', color='red')

        if save_path is not None:
            plt.savefig(save_path)
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
        raise ValueError('No files found in {}'.format(run_dir))

    if substring + '_combined.csv' in files:
        warnings.warn('Removing previously generated '
                      'combined {} file in {}'.format(substring, run_dir))
        os.remove(os.path.join(run_dir, substring + '_combined.csv'))
        files = [file for file in files if 'combined' not in file]

    # collect all metrics files
    metrics = []
    for file in files:
        metrics.append(pd.read_csv(os.path.join(run_dir, file)))

    # check that all files are the same length
    if len(metrics) > 1:
        base_len = len(metrics[0])
        for i in range(1, len(metrics)):
            if len(metrics[i]) != base_len:
                raise ValueError('Not all {} files are the same length: file {} does not match'
                                 'file {}'.format(substring, files[0], files[i]))

    metrics = pd.concat(metrics)

    metrics.to_csv(os.path.join(run_dir, substring + '_combined.csv'), index=False)


def combine_tuning_curve_metrics(dir_list):
    """Combines metrics together into a single dataframe for fitting a turning curve

    Args:
        dir_list (list): list of directories to pull metrics from

    Returns:
        pd.DataFrame: dataframe containing aggregates metrics"""

    # create list to hold all extracted data
    all_dirs = []

    # loop through each run folder
    for dir in dir_list:

        # combine tables together
        pulse_heights = pd.read_csv(os.path.join(dir, 'fov-1-scan-1_pulse_heights.csv'))
        channel_counts = pd.read_csv(os.path.join(dir, 'fov-1-scan-1_channel_counts.csv'))
        combined = pulse_heights.merge(channel_counts, 'outer', on=['fov', 'mass'])

        if len(combined) != len(pulse_heights):
            raise ValueError("Pulse heights and channel counts must be generated for the same "
                             "mass ranges and fovs. However, the data the following does do not "
                             "exactly match: {}".format(dir))

        # add directory label and add to list
        combined['directory'] = dir
        all_dirs.append(combined)

    # combine data from each dir together
    all_data = pd.concat(all_dirs)
    all_data.reset_index()

    # create normalized counts column
    subset = all_data[['channel_count', 'mass']]
    all_data['norm_channel_count'] = subset.groupby('mass').transform(lambda x: (x / x.max()))

    return all_data


def fit_mass_mph_curve(mph_vals, mass, save_dir, obj_func, min_obs=5):
    """Fits a curve for the MPH over time for the specified mass

    Args:
        mph_vals (list): mph for each FOV in the run
        mass (str or int): the mass being fit
        save_dir (str): the directory to save the fit parameters
        obj_func (str): the function to use for constructing the fit
        min_obs (int): the minimum number of observations to fit a curve, otherwise uses median"""

    fov_order = np.linspace(0, len(mph_vals) - 1, len(mph_vals))
    save_path = os.path.join(save_dir, str(mass) + '_mph_fit.jpg')

    if len(mph_vals) > min_obs:
        # fit standard curve
        weights = fit_calibration_curve(x_vals=fov_order, y_vals=mph_vals, obj_func=obj_func,
                                        plot_fit=True, save_path=save_path)
    else:
        # default to using the median instead for short runs with small number of FOVs
        mph_median = np.median(mph_vals)
        if obj_func == 'poly_2':
            weight_len = 3
        elif obj_func == 'poly_3':
            weight_len = 4
        else:
            raise ValueError("Unsupported objective function provided: {}".format(obj_func))

        # plot median
        plt.axhline(y=mph_median, color='r', linestyle='-')
        plt.plot(fov_order, mph_vals, '.')
        plt.savefig(save_path)
        plt.close()

        # all coefficients except intercept are 0
        weights = np.zeros(weight_len)
        weights[-1] = mph_median

    mass_json = {'name': obj_func, 'weights': weights.tolist()}
    mass_path = os.path.join(save_dir, str(mass) + '_norm_func.json')

    with open(mass_path, 'w') as mp:
        json.dump(mass_json, mp)


def create_fitted_mass_mph_vals(pulse_height_df, obj_func_dir):
    """Uses the mph curves for each mass to generate a smoothed mph estimate

    Args:
        pulse_height_df (pd.DataFrame): contains the MPH value per mass for all FOVs
        obj_func_dir (str): directory containing the curves generated for each mass

    Returns:
        pd.DataFrame: updated dataframe with fitted version of each MPH value for each mass"""

    # get all masses
    masses = np.unique(pulse_height_df['mass'].values)

    # create column to hold fitted values
    pulse_height_df['pulse_height_fit'] = 0

    for mass in masses:
        # load channel-specific prediction function
        mass_path = os.path.join(obj_func_dir, str(mass) + '_norm_func.json')

        with open(mass_path, 'r') as mp:
            mass_json = json.load(mp)

        name, weights = mass_json['name'], mass_json['weights']

        pred_func = create_prediction_function(name=name, weights=weights)

        # compute predicted MPH
        mass_idx = pulse_height_df['mass'] == mass
        raw_vals = pulse_height_df.loc[mass_idx, 'pulse_height'].values
        pred_vals = pred_func(raw_vals)

        # update df
        pulse_height_df.loc[mass_idx, 'pulse_height_fit'] = pred_vals

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
    masses = panel_info['Mass'].values
    fit_dir = os.path.join(norm_dir, 'curve_fits')
    os.makedirs(fit_dir)

    # combine fov-level files together
    combine_run_metrics(run_dir=pulse_height_dir, substring='pulse_heights')
    pulse_height_df = pd.read_csv(os.path.join(pulse_height_dir, 'pulse_heights_combined.csv'))

    # order by FOV
    ordering = ns.natsorted((pulse_height_df['fov'].unique()))
    pulse_height_df['fov'] = pd.Categorical(pulse_height_df['fov'],
                                            ordered=True,
                                            categories=ordering)
    pulse_height_df = pulse_height_df.sort_values('fov')

    # loop over each mass, and fit a curve for MPH over the course of the run
    for mass in masses:
        mph_vals = pulse_height_df.loc[pulse_height_df['mass'] == mass, 'pulse_height'].values
        fit_mass_mph_curve(mph_vals=mph_vals, mass=mass, save_dir=fit_dir,
                           obj_func=mass_obj_func)

    # update pulse_height_df to include fitted mph values
    pulse_height_df = create_fitted_mass_mph_vals(pulse_height_df=pulse_height_df,
                                                  obj_func_dir=fit_dir)

    return pulse_height_df


def normalize_fov(img_data, norm_vals, norm_dir, fov, channels, extreme_vals):
    """Normalize a single FOV with provided normalization constants for each channel"""

    # create directory to hold normalized images
    output_fov_dir = os.path.join(norm_dir, fov)
    if os.path.exists(output_fov_dir):
        print("output directory {} already exists, "
              "data will be overwritten".format(output_fov_dir))
    else:
        os.makedirs(output_fov_dir)

    # check if any values are outside expected range
    extreme_mask = np.logical_or(norm_vals < extreme_vals[0], norm_vals > extreme_vals[1])
    if np.any(extreme_mask):
        bad_channels = np.array(channels)[extreme_mask]
        warnings.warn('The following channel(s) had an extreme normalization '
                      'value for fov {}. Manual inspection for accuracy is '
                      'recommended: {}'.format(fov, bad_channels))

    # correct images and save
    normalized_images = img_data / norm_vals.reshape((1, 1, 1, len(norm_vals)))

    for idx, chan in enumerate(channels):
        io.imsave(os.path.join(output_fov_dir, chan + '.tiff'),
                  normalized_images[0, :, :, idx], check_contrast=False)

    # save logs
    log_df = pd.DataFrame({'channels': channels,
                           'norm_vals': norm_vals})
    log_df.to_csv(os.path.join(output_fov_dir, 'normalization_coefs.csv'), index=False)


def normalize_image_data(img_dir, norm_dir, pulse_height_dir, panel_info,
                         img_sub_folder='', mass_obj_func='poly_3', extreme_vals=(0.4, 1.1),
                         norm_func_path=os.path.join('..', 'toffy', 'norm_func.json')):
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
        raise ValueError("No normalization function found. You will need to run "
                         "section 3 of the 1_set_up_toffy.ipynb notebook to generate the "
                         "necessary function before you can normalize your data")

    # create normalization function for mapping MPH to counts
    with open(norm_func_path, 'r') as cf:
        norm_json = json.load(cf)

    img_fovs = io_utils.list_folders(img_dir, 'fov')

    norm_weights, norm_name = norm_json['weights'], norm_json['name']
    norm_func = create_prediction_function(norm_name, norm_weights)

    # combine pulse heights together into single df
    pulse_height_df = create_fitted_pulse_heights_file(pulse_height_dir=pulse_height_dir,
                                                       panel_info=panel_info, norm_dir=norm_dir,
                                                       mass_obj_func=mass_obj_func)
    channels = panel_info['Target']
    pulse_fovs = np.unique(pulse_height_df['fov'])

    # make sure FOVs used to construct tuning curve are same ones being normalized
    misc_utils.verify_same_elements(image_data_fovs=img_fovs, pulse_height_csv_files=pulse_fovs)

    # loop over each fov
    for fov in img_fovs:
        # get images and pulse heights
        images = load_utils.load_imgs_from_tree(img_dir, fovs=[fov], channels=channels,
                                                dtype='float32', img_sub_folder=img_sub_folder)

        # predict normalization based on MPH value for all masses
        pulse_height_fov = pulse_height_df.loc[pulse_height_df['fov'] == fov, :]
        norm_vals = norm_func(pulse_height_fov['pulse_height_fit'].values)

        # normalize and save
        normalize_fov(img_data=images, norm_vals=norm_vals, norm_dir=norm_dir, fov=fov,
                      channels=channels, extreme_vals=extreme_vals)
