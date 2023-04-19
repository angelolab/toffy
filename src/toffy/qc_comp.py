import copy
import itertools
import os
import pathlib
import re
from shutil import rmtree
from typing import Dict, List, Optional, Union

import natsort as ns
import numpy as np
import pandas as pd
import seaborn as sns
from alpineer import image_utils, io_utils, load_utils, misc_utils
from pandas.core.groupby import DataFrameGroupBy
from requests.exceptions import HTTPError
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

from toffy import settings
from toffy.mibitracker_utils import MibiRequests, MibiTrackerError


def create_mibitracker_request_helper(email, password):
    """Create a mibitracker request helper to access a user's MIBItracker info on Ionpath

    Args:
        email (str):
            The user's MIBItracker email address
        password (str):
            The user's MIBItracker password

    Returns:
        toffy.mibi.mibitracker_utils.MibiRequests:
            A request helper module instance to access a user's MIBItracker info
    """

    try:
        return MibiRequests(settings.MIBITRACKER_BACKEND, email, password)
    except HTTPError:
        print("Invalid MIBItracker email or password provided")


def download_mibitracker_data(
    email,
    password,
    run_name,
    run_label,
    base_dir,
    tiff_dir,
    overwrite_tiff_dir=False,
    img_sub_folder=None,
    fovs=None,
    channels=None,
):
    """Download a specific run's image data off of MIBITracker
    in an `ark` compatible directory structure

    Args:
        email (str):
            The user's MIBItracker email address
        password (str):
            The user's MIBItracker password
        run_name (str):
            The name of the run (specified on the user's MIBItracker run page)
        run_label (str):
            The label of the run (specified on the user's MIBItracker run page)
        base_dir (str):
            Where to place the created `tiff_dir`
        overwrite_tiff_dir (bool):
            Whether to overwrite the data already in `tiff_dir`
        tiff_dir (str):
            The name of the data directory in `base_dir` to write the run's image data to
        img_sub_folder (str):
            If specified, the subdirectory inside each FOV folder in `data_dir` to place
            the image data into
        fovs (list):
            A list of FOVs to subset over. If `None`, uses all FOVs.
        channels (lsit):
            A list of channels to subset over. If `None`, uses all channels.

    Returns:
        list:
            A list of tuples containing (point name, point id), sorted by point id.
            This defines the run acquisition order needed for plotting the QC graphs.
    """

    # verify that base_dir provided exists
    if not os.path.exists(base_dir):
        raise FileNotFoundError("base_dir %s does not exist" % base_dir)

    # create the MIBItracker request helper
    mr = create_mibitracker_request_helper(email, password)

    # get the run info using the run_name and the run_label
    # NOTE: there will only be one entry in the 'results' key with run_name and run_label specified
    run_info = mr.search_runs(run_name, run_label)

    # if no results are returned, invalid run_name and/or run_label provided
    if len(run_info["results"]) == 0:
        raise ValueError("No data found for run_name %s and run_label %s" % (run_name, run_label))

    # extract the name of the FOVs and their associated internal IDs
    run_fov_names = [img["number"] for img in run_info["results"][0]["imageset"]["images"]]
    run_fov_ids = [img["id"] for img in run_info["results"][0]["imageset"]["images"]]

    # if fovs is None, ensure all of the fovs in run_fov_names_ids are chosen
    if fovs is None:
        fovs = run_fov_names

    # ensure all of the fovs are valid (important if the user explicitly specifies fovs)
    misc_utils.verify_in_list(provided_fovs=fovs, mibitracker_run_fovs=run_fov_names)

    # iterate through each fov and get the longest set of channels
    # this is to prevent getting no channels if the first FOV is a Moly point or is incomplete
    image_data = run_info["results"][0]["imageset"]["images"]
    _, run_channels = max(
        {i: image_data[i]["pngs"] for i in np.arange(len(image_data))}.items(),
        key=lambda x: len(set(x[1])),
    )

    # if channels is None, ensure all of the channels in run_channels are chosen
    if channels is None:
        channels = run_channels

    # ensure all of the channels are valid (important if the user explicitly specifies channels)
    misc_utils.verify_in_list(provided_chans=channels, mibitracker_run_chans=run_channels)

    # if the desired tiff_dir exists, remove it if overwrite_tiff_dir is True
    # otherwise, throw an error
    if os.path.exists(os.path.join(base_dir, tiff_dir)):
        if overwrite_tiff_dir:
            print("Overwriting existing data in tiff_dir %s" % tiff_dir)
            rmtree(os.path.join(base_dir, tiff_dir))
        else:
            raise ValueError("tiff_dir %s already exists in %s" % (tiff_dir, base_dir))

    # make the image directory
    os.mkdir(os.path.join(base_dir, tiff_dir))

    # ensure sub_folder gets set to "" if img_sub_folder is None (for os.path.join convenience)
    if not img_sub_folder:
        img_sub_folder = ""

    # define the run order list to return
    run_order = []

    # iterate over each FOV of the run
    for img in run_info["results"][0]["imageset"]["images"]:
        # if the image fov name is not specified, move on
        if img["number"] not in fovs:
            continue

        print("Creating data for fov %s" % img["number"])

        # make the fov directory
        os.mkdir(os.path.join(base_dir, tiff_dir, img["number"]))

        # make the img_sub_folder inside the fov directory if specified
        if len(img_sub_folder) > 0:
            os.mkdir(os.path.join(base_dir, tiff_dir, img["number"], img_sub_folder))

        # iterate over each provided channel
        for chan in channels:
            # extract the channel data from MIBItracker as a numpy array
            # fail on the whole FOV if the channel is not found (most likely a Moly point)
            try:
                chan_data = mr.get_channel_data(img["id"], chan)
            except MibiTrackerError as mte:
                print(
                    "On FOV %s, failed to download channel %s, moving on to the next FOV. "
                    "If FOV %s is a Moly point, ignore. "
                    "Otherwise, please ensure that channel %s exists on MibiTracker for FOV %s"
                    % (img["number"], chan, img["number"], chan, img["number"])
                )

                # clean the FOV: we will not have a folder for it (in case of Moly point)
                rmtree(os.path.join(base_dir, tiff_dir, img["number"]))

                # do not attempt to download any more channels
                break

            # define the name of the channel file
            chan_file = "%s.tiff" % chan

            # write the data to a .tiff file in the FOV directory structure
            fname: str = os.path.join(base_dir, tiff_dir, img["number"], img_sub_folder, chan_file)
            image_utils.save_image(fname, chan_data)

        # append the run name and run id to the list
        run_order.append((img["number"], img["id"]))

    return run_order


def compute_nonzero_mean_intensity(image_data):
    """Compute the nonzero mean of a specific fov/chan pair

    Args:
        image_data (numpy.ndarray):
            the image data for a specific fov/chan pair

    Returns:
        float:
            The nonzero mean intensity of the fov/chan pair (`np.nan` if channel contains all 0s)
    """

    # take just the non-zero pixels
    image_data_nonzero = image_data[image_data != 0]

    # take the mean of the non-zero pixels and assign to (fov, channel) in array
    # unless there are no non-zero pixels, in which case default to 0
    if len(image_data_nonzero) > 0:
        nonzero_mean_intensity = image_data_nonzero.mean()
    else:
        nonzero_mean_intensity = 0

    return nonzero_mean_intensity


def compute_total_intensity(image_data):
    """Compute the sum of all pixels of a specific fov/chan pair

    Args:
        image_data (numpy.ndarray):
            the image data for a specific fov/chan pair

    Returns:
        float:
            The total intensity of the fov/chan pair (`np.nan` if channel contains all 0s)
    """

    return np.sum(image_data)


def compute_99_9_intensity(image_data):
    """Compute the 99.9% pixel intensity value of a specific fov/chan pair

    Args:
        image_data (numpy.ndarray):
            the image data for a specific fov/chan pair

    Returns:
        float:
            The 99.9% pixel intensity value of a specific fov/chan pair
    """

    return np.percentile(image_data, q=99.9)


def sort_bin_file_fovs(fovs, suffix_ignore=None):
    """Sort a list of fovs in a bin file by fov and scan number

    fovs (list):
        a list of fovs prefixed with `'fov-m-scan-n'`
    suffix_ignore (str):
        removes this at the end of each fov name, needed if sorting fov-level QC `.csv` files

    Returns:
        list:
            fov name list sorted by ascending fov number, the ascending scan number
    """

    # set suffix_ignore to the empty string if None
    if suffix_ignore is None:
        suffix_ignore = ""

    # TODO: if anyone can do this using a walrus operator I'd appreciate it!
    return sorted(
        fovs,
        key=lambda f: (
            int(f.replace(suffix_ignore, "").split("-")[1]),
            int(f.replace(suffix_ignore, "").split("-")[3]),
        ),
    )


def compute_qc_metrics(
    extracted_imgs_path, fov_name, gaussian_blur=False, blur_factor=1, save_csv=None
):
    """Compute the QC metric matrices for the image data provided

    Args:
        extracted_imgs_path (str):
            the directory where extracted images are stored
        fov_name (str):
            the name of the FOV to extract from `bin_file_path`, needs to correspond with JSON name
        gaussian_blur (bool):
            whether or not to add Gaussian blurring
        blur_factor (int):
            the sigma (standard deviation) to use for Gaussian blurring
            set to 0 to use raw inputs without Gaussian blurring
            ignored if `gaussian_blur` set to `False`
        save_csv (str):
            path to save csvs of the qc metrics to

    Returns:
        None
    """

    # path validation checks
    if not os.path.exists(extracted_imgs_path):
        raise FileNotFoundError("extracted_imgs_path %s does not exist" % extracted_imgs_path)

    # retrieve the image data from extracted tiff files
    # the image coords should be: ['fov', 'type', 'x', 'y', 'channel']
    image_data = load_utils.load_imgs_from_tree(extracted_imgs_path, fovs=[fov_name])
    image_data = format_img_data(image_data)

    metric_csvs = compute_qc_metrics_direct(image_data, fov_name, gaussian_blur, blur_factor)
    if save_csv:
        for metric_name, data in metric_csvs.items():
            data.to_csv(os.path.join(save_csv, metric_name), index=False)


def compute_qc_metrics_direct(image_data, fov_name, gaussian_blur=False, blur_factor=1):
    """Compute the QC metric matrices for the image data provided

    Args:
        image_data (xr.DataArray):
            image data in 'extract_bin_files' output format
        fov_name (str):
            the name of the FOV to extract from `bin_file_path`, needs to correspond with JSON name
        gaussian_blur (bool):
            whether or not to add Gaussian blurring
        blur_factor (int):
            the sigma (standard deviation) to use for Gaussian blurring
            set to 0 to use raw inputs without Gaussian blurring
            ignored if `gaussian_blur` set to `False`

    """

    # there's only 1 FOV and 1 type ('pulse'), so subset on that
    image_data = image_data.loc[fov_name, "pulse", :, :, :]

    # define the list of channels to use
    chans = image_data.channel.values

    # define numpy arrays for all the metrics to extract, more efficient indexing than pandas
    blank_arr = np.zeros(image_data.shape[2], dtype="float32")
    nonzero_mean_intensity = copy.deepcopy(blank_arr)
    total_intensity = copy.deepcopy(blank_arr)
    intensity_99_9 = copy.deepcopy(blank_arr)

    # it's faster to loop through the individual channels rather than broadcasting
    for i, chan in enumerate(chans):
        # subset on the channel, cast to float32 to prevent truncation
        image_data_np = image_data.loc[:, :, chan].values.astype(np.float32)

        # STEP 1: gaussian blur (if specified)
        if gaussian_blur:
            image_data_np = gaussian_filter(
                image_data_np, sigma=blur_factor, mode="nearest", truncate=2.0
            )

        # STEP 2: extract non-zero mean intensity
        nonzero_mean_intensity[i] = compute_nonzero_mean_intensity(image_data_np)

        # STEP 3: extract total intensity
        total_intensity[i] = compute_total_intensity(image_data_np)

        # STEP 4: take 99.9% value of the data and assign
        intensity_99_9[i] = compute_99_9_intensity(image_data_np)

    # define the list of numpy arrays for looping
    metric_data = [nonzero_mean_intensity, total_intensity, intensity_99_9]

    metric_csvs = {}

    for ms, md, mc in zip(settings.QC_SUFFIXES, metric_data, settings.QC_COLUMNS):
        # define the dataframe for this metric
        metric_df = pd.DataFrame(columns=["fov", "channel", mc], dtype=object)

        # assign the metric data
        metric_df[mc] = md

        # assign the fov and channel names
        metric_df["fov"] = fov_name
        metric_df["channel"] = chans

        metric_csvs[f"{fov_name}_{ms}.csv"] = metric_df

    return metric_csvs


def combine_qc_metrics(qc_metrics_dir):
    """Aggregates the QC results of each FOV into one `.csv`

    Args:
        qc_metrics_dir (str):
            the name of the folder containing the QC metric files
    """

    # path validation check
    if not os.path.exists(qc_metrics_dir):
        raise FileNotFoundError("qc_metrics_dir %s does not exist" % qc_metrics_dir)

    for ms in settings.QC_SUFFIXES:
        # define an aggregated metric DataFrame
        metric_df = pd.DataFrame()

        # list all the files corresponding to this metric
        metric_files = io_utils.list_files(qc_metrics_dir, substrs=ms + ".csv")

        # don't consider any existing combined .csv files, just the fov-level .csv files
        metric_files = [mf for mf in metric_files if "combined" not in mf]

        # sort the files to ensure consistency
        metric_files = sort_bin_file_fovs(metric_files, suffix_ignore="_%s.csv" % ms)

        # iterate over each metric file and append the data to metric_df
        for mf in metric_files:
            metric_df = pd.concat([metric_df, pd.read_csv(os.path.join(qc_metrics_dir, mf))])

        # write the aggregated metric data
        # NOTE: if this combined metric file already exists, it will be overwritten
        metric_df.to_csv(os.path.join(qc_metrics_dir, "combined_%s.csv" % ms), index=False)


def visualize_qc_metrics(
    metric_name: str,
    qc_metric_dir: Union[str, pathlib.Path],
    save_dir: Union[str, pathlib.Path],
    channel_filters: Optional[List[str]] = ["chan_"],
    axes_font_size: int = 16,
    wrap: int = 6,
    dpi: int = 300,
    return_plot: bool = False,
) -> Optional[sns.FacetGrid]:
    """
    Visualize the barplot of a specific QC metric.

    Args:
        metric_name (str):
            The name of the QC metric to plot. Used as the y-axis label. Options include:
            `"Non-zero mean intensity"`, `"Total intensity"`, `"99.9% intensity value"`.
        qc_metric_dir (Union[str, pathlib.Path]):
            The path to the directory containing the `'combined_{qc_metric}.csv'` files
        save_dir (Optional[Union[str, pathlib.Path]], optional):
            The name of the directory to save the plot to. Defaults to None.
        channel_filters (List[str], optional):
            A list of channels to filter out.
        axes_font_size (int, optional):
            The font size of the axes labels. Defaults to 16.
        wrap (int, optional):
            The number of plots to display per row. Defaults to 6.
        dpi (Optional[int], optional):
            The resolution of the image to use for saving. Defaults to None.
        return_plot (bool):
            If `True`, this will return the plot. Defaults to `False`

    Raises:
        ValueError:
            When an invalid metric is provided.
        FileNotFoundError:
            The QC metric directory `qc_metric_dir` does not exist.
        FileNotFoundError:
            The QC metric `combined_csv` file is does not exist in `qc_metric_dir`.

    Returns:
        Optional[sns.FacetGrid]: Returns the Seaborn FacetGrid catplot of the QC metrics.
    """
    # verify the metric provided is valid
    if metric_name not in settings.QC_COLUMNS:
        raise ValueError(
            "Invalid metric %s provided, must be set to 'Non-zero mean intensity', "
            "'Total intensity', or '99.9%% intensity value'" % metric_name
        )

    # verify the path to the QC metric datasets exist
    if not os.path.exists(qc_metric_dir):
        raise FileNotFoundError("qc_metric_dir %s does not exist" % qc_metric_dir)

    # get the file name of the combined QC metric .csv file to use
    qc_metric_index = settings.QC_COLUMNS.index(metric_name)
    qc_metric_suffix = settings.QC_SUFFIXES[qc_metric_index]
    qc_metric_path = os.path.join(qc_metric_dir, "combined_%s.csv" % qc_metric_suffix)

    # ensure the user set the right qc_metric_dir
    if not os.path.exists(qc_metric_path):
        raise FileNotFoundError(
            "Could not locate %s, ensure qc_metric_dir is correct" % qc_metric_path
        )

    # read in the QC metric data
    qc_metric_df = pd.read_csv(qc_metric_path)

    # filter out naturally-occurring elements as well as Noodle
    qc_metric_df = qc_metric_df[~qc_metric_df["channel"].isin(settings.QC_CHANNEL_IGNORE)]

    # filter out any channel in the channel_filters list
    if channel_filters is not None:
        qc_metric_df: pd.DataFrame = qc_metric_df[
            ~qc_metric_df["channel"].str.contains("|".join(channel_filters))
        ]

    # catplot allows for easy facets on a barplot
    qc_fg: sns.FacetGrid = sns.catplot(
        x="fov",
        y=metric_name,
        col="channel",
        col_wrap=wrap,
        data=qc_metric_df,
        kind="bar",
        color="black",
        sharex=True,
        sharey=False,
    )

    # remove the 'channel =' in each subplot title
    qc_fg.set_titles(template="{col_name}")
    qc_fg.figure.supxlabel(t="fov", x=0.5, y=0, ha="center", size=axes_font_size)
    qc_fg.figure.supylabel(t=f"{metric_name}", x=0, y=0.5, va="center", size=axes_font_size)
    qc_fg.set(xticks=[], yticks=[])

    # per Erin's visualization remove the default axis title on the y-axis
    # and instead show 'fov' along x-axis and the metric name along the y-axis (overarching)
    qc_fg.set_axis_labels(x_var="", y_var="")
    qc_fg.set_xticklabels([])
    qc_fg.set_yticklabels([])

    # save the figure always
    # Return the figure if specified.
    qc_fg.savefig(os.path.join(save_dir, f"{metric_name}_barplot_stats.png"), dpi=dpi)

    if return_plot:
        return qc_fg


def format_img_data(img_data):
    """Formats the image array from load_imgs_from_tree to be same structure as the array returned
    by extract_bin_files. Works for one FOV data at a time.
    Args:
        img_data (str): current image data array as produced by load function
    Returns:
         xarray.DataArray: image data array with shape [fov, type, x, y, channel]
    """

    # add type dimension
    img_data = img_data.assign_coords(type="pulse")
    img_data = img_data.expand_dims("type", 1)

    # edit dimension names
    img_data = img_data.rename({"fovs": "fov", "rows": "x", "cols": "y", "channels": "channel"})

    return img_data


def _get_r_c(fov_name: pd.Series, search_term: re.Pattern) -> pd.Series:
    """Gets the row and column value from a FOV's name containing RnCm.

    Args:
        fov_name (pd.Series): The FOV's name.
        search_term (re.Pattern): The regex pattern for searching for RnCm.

    Returns:
        pd.Series: Returns `n` and `m` as a series.
    """
    r, c = map(int, re.search(search_term, fov_name).group(1, 2))
    return pd.Series([r, c])


def qc_tma_metrics(
    extracted_imgs_path: Union[str, pathlib.Path],
    qc_tma_metrics_dir: Union[str, pathlib.Path],
    tma: str,
) -> None:
    """
    Calculates the QC metrics for a user specified TMA.

    Args:
        extracted_imgs_path (Union[str,pathlib.Path]): The directory where the extracted images are stored.
        qc_tma_metrics_dir (Union[str, pathlib.path]): The directory where to place the QC TMA metrics.
        tma (str): The FOVs with the TMA in the folder name to gather.
    """
    # Get all the FOVs that match the input `tma` string
    fovs = io_utils.list_folders(extracted_imgs_path, substrs=tma)

    # Create regex pattern for searching RnCm
    search_term: re.Pattern = re.compile(r"R\+?(\d+)C\+?(\d+)")

    # Get qc metrics for each fov
    for fov in ns.natsorted(fovs):
        compute_qc_metrics(
            extracted_imgs_path=extracted_imgs_path, fov_name=fov, save_csv=qc_tma_metrics_dir
        )

    # Combine the qc metrics for all fovs per TMA
    for ms in settings.QC_SUFFIXES:
        metric_files: List[str] = io_utils.list_files(qc_tma_metrics_dir, substrs=f"{ms}.csv")
        metric_files: List[str] = [mf for mf in metric_files if "combined" not in mf]

        # Define an aggregated metric DataFrame
        combined_metric_df: pd.DataFrame = pd.concat(
            (pd.read_csv(os.path.join(qc_tma_metrics_dir, mf)) for mf in metric_files),
            ignore_index=True,
        )

        # Extract the Row and Column
        combined_metric_df[["row", "column"]] = combined_metric_df["fov"].apply(
            lambda row: _get_r_c(row, search_term)
        )
        combined_metric_df.to_csv(
            os.path.join(qc_tma_metrics_dir, f"{tma}_combined_{ms}.csv"), index=False
        )


def _create_r_c_tma_matrix(
    group: DataFrameGroupBy, x_size: int, y_size: int, qc_col: str
) -> pd.Series:
    """
    Creates the FOV / TMA matrix.

    Args:
        group (DataFrameGroupBy): Each group consists of an individual channel, and all of it's associated FOVs.
        x_size (int): The number of columns in the matrix.
        y_size (int): The number of rows in the matrix.
        qc_col (str): The column to get the the QC data.

    Returns:
        pd.Series[np.ndarray]: Returns the a series containing the matrix.
    """

    rc_array: np.ndarray = np.full(shape=(x_size, y_size), fill_value=np.nan)
    rc_array[group["column"] - 1, group["row"] - 1] = group[qc_col]

    return pd.Series([rc_array])


def qc_tma_metrics_rank(
    qc_tma_metrics_dir: Union[str, pathlib.Path],
    tma: str,
    qc_metrics: List[str] = None,
    channel_exclude: List[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Creates the average rank for a given TMA across all FOVs and unfiltered / unexcluded channels.
    By default the following channels are excluded: Au, Fe, Na, Ta, Noodle.


    Args:
        qc_tma_metrics_dir (Union[str, pathlib.Path]): The direcftory where to place the QC TMA metrics.
        tma (str): The TMA to gather FOVs in.
        qc_metrics (List[str], optional): The QC metrics to create plots for. Can be a subset of the
        following:

            * Non-zero mean intensity
            * Total intensity
            * 99.9% intensity value. Defaults to None.
        channel_exclude (List[str], optional): An optional list of channels to further filter out. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the QC column and the a numpy array
        representing the average ranks for a given TMA."""
    # Sort the loaded combined csv files based on QC_SUFFIXES
    combined_metric_tmas = ns.natsorted(
        io_utils.list_files(qc_tma_metrics_dir, substrs=f"{tma}_combined"),
        key=lambda m: (i for i, qc_s in enumerate(settings.QC_SUFFIXES) if qc_s in m),
    )
    # Then filter out unused suffixes
    if qc_metrics is not None:
        filtered_qcs: List[bool] = [qcm in qc_metrics for qcm in settings.QC_COLUMNS]
        qc_cols = list(itertools.compress(settings.QC_COLUMNS, filtered_qcs))
        combined_metric_tmas = list(itertools.compress(combined_metric_tmas, filtered_qcs))
    else:
        qc_cols: List[str] = settings.QC_COLUMNS

    cmt_data = dict()
    for cmt, qc_col in zip(combined_metric_tmas, qc_cols):
        # Open and filter the default ignored channels
        cmt_df: pd.DataFrame = pd.read_csv(os.path.join(qc_tma_metrics_dir, cmt))
        cmt_df: pd.DataFrame = cmt_df[~cmt_df["channel"].isin(settings.QC_CHANNEL_IGNORE)]

        # Verify that the excluded channels exist in the combined metric tma DataFrame
        # Then remove the excluded channels
        if channel_exclude is not None:
            misc_utils.verify_in_list(
                channels_to_exclude=channel_exclude,
                combined_metric_tma_df_channels=cmt_df["channel"].unique(),
            )
            cmt_df: pd.DataFrame = cmt_df[~cmt_df["channel"].isin(channel_exclude)]

        # Get matrix dimensions
        y_size: int = cmt_df["column"].max()
        x_size: int = cmt_df["row"].max()

        # Create the TMA matrix / for the heatmap
        channel_tmas: pd.DataFrame = cmt_df.groupby(by="channel", sort=True).apply(
            lambda group: _create_r_c_tma_matrix(group, y_size, x_size, qc_col)
        )
        channel_matrices: np.ndarray = np.array(
            [c_tma[0] for c_tma in channel_tmas.values],
        )

        # Rank all FOVs for each channel.
        ranked_channels: np.ndarray = rankdata(
            a=channel_matrices.reshape((x_size * y_size), -1),
            method="average",
            nan_policy="omit",
            axis=0,
        ).reshape(len(channel_tmas), x_size, y_size)

        # Average the rank for each channel.
        avg_ranked_tma: np.ndarray = ranked_channels.mean(axis=0)

        cmt_data[qc_col] = avg_ranked_tma

    return cmt_data


def batch_effect_qc_metrics(
    cohort_data_dir: Union[str, pathlib.Path],
    qc_cohort_metrics_dir: Union[str, pathlib.Path],
    tissues: List[str],
) -> None:
    """
    Computes QC metrics for a specified set of tissues and saves the tissue specific QC files
    in the `qc_cohort_metrics_dir`. Calculates the following metrics for the specified tissues,
    and the metrics for the invidual FOVs within that cohort:
    * Non-zero mean intensity
    * Total intensity
    * 99.9% intensity value

    Args:
        cohort_data_dir (Union[str, pathlib.Path]): The directory which contains the FOVs for a cohort of interest.
        qc_cohort_metrics_dir (Union[str,pathlib.Path]): The directory where the cohort metrics will be saved to.
        tissues (List[str]): A list of tissues to find QC metrics for.

    Raises:
        ValueError: Errors if `tissues` is either None, or a list of size 0.
    """
    if tissues is None or len(tissues) < 1:
        raise ValueError("The tissues must be specified")

    # Input validation: cohort_data_dir, qc_cohort_metrics_dir
    io_utils.validate_paths([cohort_data_dir, qc_cohort_metrics_dir])

    samples = io_utils.list_folders(dir_name=cohort_data_dir, substrs=tissues)

    tissue_to_sample_mapping: Dict[str, List[str]] = {}

    for sample, tissue in itertools.product(samples, tissues):
        if tissue in sample:
            if tissue not in tissue_to_sample_mapping.keys():
                tissue_to_sample_mapping[tissue] = [sample]
            else:
                tissue_to_sample_mapping[tissue].append(sample)

    # Use a set of the samples to avoid duplicate QC metric calculations
    sample_set = set(list(itertools.chain.from_iterable(tissue_to_sample_mapping.values())))

    # Compute the QC metrics for all unique samples that match with the user's tissue input.
    for sample in ns.natsorted(sample_set):
        compute_qc_metrics(
            extracted_imgs_path=cohort_data_dir, fov_name=sample, save_csv=qc_cohort_metrics_dir
        )

    # Combined metrics per Tissue
    for (tissue, samples), ms in itertools.product(
        tissue_to_sample_mapping.items(), settings.QC_SUFFIXES
    ):
        metric_files: List[str] = io_utils.list_files(
            qc_cohort_metrics_dir, substrs=[f"{sample}_{ms}.csv" for sample in samples]
        )

        metric_files = list(filter(lambda mf: "combined" not in mf, metric_files))

        # Define an aggregated metric DataFrame
        combined_metric_tissue_df: pd.DataFrame = pd.concat(
            (pd.read_csv(os.path.join(qc_cohort_metrics_dir, mf)) for mf in metric_files)
        )

        combined_metric_tissue_df.to_csv(
            os.path.join(qc_cohort_metrics_dir, f"{tissue}_combined_{ms}.csv"),
            index=False,
        )
