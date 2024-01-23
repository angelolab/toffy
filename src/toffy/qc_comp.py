import copy
import itertools
import os
import pathlib
import re
import warnings
from dataclasses import dataclass, field
from shutil import rmtree
from typing import Dict, List, Optional, Tuple, Union

import natsort as ns
import numpy as np
import pandas as pd
import xarray as xr
from alpineer import image_utils, io_utils, load_utils, misc_utils
from pandas.core.groupby import DataFrameGroupBy
from requests.exceptions import HTTPError
from scipy import stats
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

from toffy import settings
from toffy.mibitracker_utils import MibiRequests, MibiTrackerError


def create_mibitracker_request_helper(email, password):
    """Create a mibitracker request helper to access a user's MIBItracker info on Ionpath.

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
    in an `ark` compatible directory structure.

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
            except MibiTrackerError:
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
    """Compute the nonzero mean of a specific fov/chan pair.

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
    """Compute the sum of all pixels of a specific fov/chan pair.

    Args:
        image_data (numpy.ndarray):
            the image data for a specific fov/chan pair

    Returns:
        float:
            The total intensity of the fov/chan pair (`np.nan` if channel contains all 0s)
    """
    return np.sum(image_data)


def compute_99_9_intensity(image_data):
    """Compute the 99.9% pixel intensity value of a specific fov/chan pair.

    Args:
        image_data (numpy.ndarray):
            the image data for a specific fov/chan pair

    Returns:
        float:
            The 99.9% pixel intensity value of a specific fov/chan pair
    """
    return np.percentile(image_data, q=99.9)


def sort_bin_file_fovs(fovs, suffix_ignore=None):
    """Sort a list of fovs in a bin file by fov and scan number.

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
    """Compute the QC metric matrices for the image data provided.

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
    """Compute the QC metric matrices for the image data provided.

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


def combine_qc_metrics(qc_metrics_dir, warn_overwrite=True):
    """Aggregates the QC results of each FOV into one `.csv`.

    Args:
        qc_metrics_dir (str):
            the name of the folder containing the QC metric files
        warn_overwrite (bool):
            whether to warn if existing combined CSV found for each metric
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
        if os.path.exists(os.path.join(qc_metrics_dir, "combined_%s.csv" % ms)) and warn_overwrite:
            warnings.warn(
                "Removing previously generated combined %s file in %s" % (ms, qc_metrics_dir)
            )
        metric_df.to_csv(os.path.join(qc_metrics_dir, "combined_%s.csv" % ms), index=False)


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


def qc_filtering(qc_metrics: List[str]) -> Tuple[List[str], List[str]]:
    """Filters the QC columns and suffixes based on the user specified QC metrics,
    then sorts the suffixes w.r.t the columns. Refer to `settings.py` for the
    Column and Suffixes available.

    Args:
        qc_metrics (List[str]): A list of QC metrics to use. Options include:

                - `"Non-zero mean intensity"`
                - `"Total intensity"`
                - `"99.9% intensity value"`


    Returns:
        Tuple[List[str], List[str]]: Returns the QC Columns and the QC Suffixes
    """
    # Filter out unused QC columns and suffixes
    if qc_metrics is not None:
        selected_qcs: List[bool] = [qcm in qc_metrics for qcm in settings.QC_COLUMNS]
        qc_cols = list(itertools.compress(settings.QC_COLUMNS, selected_qcs))
        qc_suffixes = list(itertools.compress(settings.QC_SUFFIXES, selected_qcs))
    else:
        qc_cols: List[str] = settings.QC_COLUMNS
        qc_suffixes: List[str] = settings.QC_SUFFIXES

    return qc_cols, qc_suffixes


def _channel_filtering(
    df: pd.DataFrame, channel_include: List[str] = None, channel_exclude: List[str] = None
) -> pd.DataFrame:
    """Filters the DataFrame based on the included and excluded channels. In addition
    the default ignored channels; Au, Fe, Na, Ta, Noodle, are removed.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        channel_include (List[str], optional): A list of channels to include. Defaults to None.
        channel_exclude (List[str], optional): A list of channels to exclude. Defaults to None.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if (
        isinstance(channel_include, list)
        and isinstance(channel_exclude, list)
        and not set(channel_exclude).isdisjoint(set(channel_include))
    ):
        raise ValueError("You cannot include and exclude the same channel.")

    # Filter out the default ignored channels
    df = df[~df["channel"].isin(settings.QC_CHANNEL_IGNORE)]

    # Remove the excluded channels
    # If a channel does not exist, it is ignored
    if channel_exclude is not None:
        df: pd.DataFrame = df[~df["channel"].isin(channel_exclude)]

    # Then filter the excluded channels
    # If a channel does not exist, it is ignored
    if channel_include is not None:
        df: pd.DataFrame = df[df["channel"].isin(channel_include)]
    return df


@dataclass
class QCTMA:
    """Computes the QC metrics for a given list of TMAs of interest and saves TMA specific QC files
    in the `qc_tma_metrics_dir` directory.

    Args:
        cohort_path (Union[str,pathlib.Path]): The directory where the extracted images are
            stored.
        qc_tma_metrics_dir (Union[str, pathlib.path]): The directory where to save the QC TMA
            metrics.
        qc_metrics (List[str]): A list of QC metrics to use. Options include:

                - `"Non-zero mean intensity"`
                - `"Total intensity"`
                - `"99.9% intensity value"`

    Attributes:
        qc_cols (List[str]): A list of the QC columns.
        qc_suffixes (List[str]): A list of the QC suffixes, ordered w.r.t `qc_cols`.
        search_term (re.Pattern): The regex pattern to extract n,m from FOV names of the form RnCm.
        tma_avg_zscores (Dict[str, xr.DataArray]): A dictionary containing the average z-scores for
            each TMA for each QC Metric in `qc_metrics`.
    """

    qc_metrics: Optional[List[str]]
    cohort_path: Union[str, pathlib.Path]
    metrics_dir: Union[str, pathlib.Path]

    # Fields initialized after `__post_init__`
    search_term: re.Pattern = field(init=False)
    qc_cols: List[str] = field(init=False)
    qc_suffixes: List[str] = field(init=False)

    # Set by methods
    tma_avg_zscores: Dict[str, xr.DataArray] = field(init=False)

    def __post_init__(self):
        """Initialize QCTMA."""
        # Input validation: Ensure that the paths exist
        io_utils.validate_paths([self.cohort_path, self.metrics_dir])

        self.qc_cols, self.qc_suffixes = qc_filtering(qc_metrics=self.qc_metrics)

        # Create regex pattern for searching RnCm
        self.search_term: re.Pattern = re.compile(r"R\+?(\d+)C\+?(\d+)")

        # Set the tma_avg_zscores to be an empty dictionary
        self.tma_avg_zscores = {}

    def _get_r_c(self, fov_name: pd.Series) -> pd.Series:
        """Extracts the row and column value from a FOV's name containing RnCm.

        Args:
            fov_name (pd.Series): The FOV's name.

        Returns:
            pd.Series: Returns `n` and `m` as a series of integers.
        """
        r, c = map(int, re.search(self.search_term, fov_name).group(1, 2))
        return pd.Series([r, c])

    def _create_r_c_tma_matrix(
        self, group: DataFrameGroupBy, n_cols: int, n_rows: int, qc_col: str
    ) -> pd.Series:
        """Z-scores all FOVS for a given channel and creates a matrix of size `n_rows` by `n_cols`
        as the TMA grid.

        Args:
            group (DataFrameGroupBy): Each group consists of an individual channel, and all of it's
                associated FOVs.
            n_cols (int): The number of columns in the matrix.
            n_rows (int): The number of rows in the matrix.
            qc_col (str): The column to get the the QC data.

        Returns:
            pd.Series[np.ndarray]: Returns the a series containing the z-score matrix.
        """
        rc_array: np.ndarray = np.full(shape=(n_cols, n_rows), fill_value=np.nan)
        rc_array[group["column"] - 1, group["row"] - 1] = stats.zscore(group[qc_col])

        return pd.Series([rc_array])

    def compute_qc_tma_metrics(self, tmas: List[str]):
        """Calculates the QC metrics for a user specified list of TMAs.

        Args:
            tmas (List[str]): The FOVs with the TMA in the folder name to gather.
        """
        with tqdm(
            total=len(tmas), desc="Computing QC TMA Metrics", unit="TMA", leave=True
        ) as tma_pbar:
            for tma in tmas:
                self._compute_qc_tma_metrics(tma=tma)
                tma_pbar.set_postfix(TMA=tma)
                tma_pbar.update(n=1)

    def _compute_qc_tma_metrics(self, tma: str):
        """Computes the FOV QC metrics for all FOVs in a given TMA.
        If the QC metrics have already been computed, then.

        Args:
            tma (str): The TMA to compute the QC metrics for.
        """
        # cannot use `io_utils.list_folders` because it cannot do a partial "exact match"
        # i.e. if we want to match `tma_1_Rn_Cm` but not `tma_10_Rn_Cm`, `io_utils.list_folders`
        # will return both for `tma_1`
        fovs: List[str] = ns.natsorted(
            seq=(p.name for p in pathlib.Path(self.cohort_path).glob(f"{tma}_*"))
        )

        # Compute the QC metrics
        with tqdm(fovs, desc="Computing QC Metrics", unit="FOV", leave=False) as pbar:
            for fov in pbar:
                # Gather the qc tma files for the current fov if they exist
                pre_computed_metrics = filter(
                    lambda f: "combined" not in f,
                    io_utils.list_files(
                        dir_name=self.metrics_dir,
                        substrs=[f"{fov}_{qc_suffix}.csv" for qc_suffix in self.qc_suffixes],
                    ),
                )

                # only compute if any QC files are missing for the current fov
                if len(list(pre_computed_metrics)) != len(self.qc_cols):
                    compute_qc_metrics(
                        extracted_imgs_path=self.cohort_path,
                        fov_name=fov,
                        save_csv=self.metrics_dir,
                    )
                    pbar.set_postfix(FOV=fov, status="Computing")
                else:
                    pbar.set_postfix(FOV=fov, status="Already Computed")

        # Generate the combined metrics for each TMA
        for qc_suffix in self.qc_suffixes:
            metric_files: List[str] = ns.natsorted(
                (
                    io_utils.list_files(
                        dir_name=self.metrics_dir,
                        substrs=[f"{fov}_{qc_suffix}.csv" for fov in fovs],
                    )
                )
            )

            # Define an aggregated metric DataFrame
            combined_metric_tissue_df: pd.DataFrame = pd.concat(
                (pd.read_csv(os.path.join(self.metrics_dir, mf)) for mf in metric_files)
            )

            combined_metric_tissue_df.to_csv(
                os.path.join(self.metrics_dir, f"{tma}_combined_{qc_suffix}.csv"),
                index=False,
            )

    def qc_tma_metrics_zscore(self, tmas: List[str], channel_exclude: List[str] = None):
        """Creates the average zscore for a given TMA across all FOVs and unexcluded channels.
        By default the following channels are excluded: Au, Fe, Na, Ta, Noodle.

        Args:
            tmas (List[str]): The FOVs withmetet the TMA in the folder name to gather.
            channel_exclude (List[str], optional): An optional list of channels to further filter
                out. Defaults to None.
        """
        max_col, max_row = 0, 0
        with tqdm(total=len(tmas), desc="Computing QC TMA Metric Z-scores", unit="TMA") as pbar:
            for tma in tmas:
                self.tma_avg_zscores[tma] = self._compute_qc_tma_metrics_zscore(
                    tma, channel_exclude=channel_exclude
                )
                max_col = max(self.tma_avg_zscores[tma].shape[1], max_col)
                max_row = max(self.tma_avg_zscores[tma].shape[2], max_row)

                pbar.set_postfix(TMA=tma)
                pbar.update()

        # also average z-scores and store
        all_tmas = np.full(
            shape=(len(tmas), len(self.qc_metrics), max_col, max_row), fill_value=np.nan
        )
        for i, tma in enumerate(tmas):
            col, row = self.tma_avg_zscores[tma].shape[1], self.tma_avg_zscores[tma].shape[2]
            all_tmas[i, :, :col, :row] = self.tma_avg_zscores[tma]

        self.tma_avg_zscores["cross_TMA_averages"] = xr.DataArray(
            data=np.stack(np.nanmean(all_tmas, axis=0)),
            coords=[self.qc_cols, np.arange(max_col), np.arange(max_row)],
            dims=["qc_col", "cols", "rows"],
        )

    def _compute_qc_tma_metrics_zscore(
        self,
        tma: str,
        channel_exclude: List[str] = None,
    ) -> xr.DataArray:
        """Creates the average z-score for a given TMA across all FOVs and unexcluded channels.
        By default the following channels are excluded: Au, Fe, Na, Ta, Noodle.

        Args:
            tma (str): The TMA to compute the average z-score for.
            channel_exclude (List[str], optional): An optional list of channels to further filter
                out. Defaults to None.

        Returns:
            xr.DataArray: An xarray DataArray containing the average z-score for each channel across
                a TMA.
        """
        # Sort the loaded combined csv files based on the filtered `qc_suffixes`
        combined_metric_tmas: List[str] = ns.natsorted(
            io_utils.list_files(self.metrics_dir, substrs=f"{tma}_combined"),
            key=lambda tma_mf: (i for i, qc_s in enumerate(self.qc_suffixes) if qc_s in tma_mf),
        )

        zscore_channels_matrix = []
        n_cols: int = None
        n_rows: int = None

        for cmt, qc_col in zip(combined_metric_tmas, self.qc_cols):
            # Open and filter the default ignored channels, along with the user specified channels
            cmt_df: pd.DataFrame = _channel_filtering(
                df=pd.read_csv(os.path.join(self.metrics_dir, cmt)), channel_exclude=channel_exclude
            )

            cmt_df[["column", "row"]] = cmt_df["fov"].apply(self._get_r_c)

            # Get matrix dimensions
            n_cols = cmt_df["column"].max()
            n_rows = cmt_df["row"].max()

            # Z-score all FOVs per channel, and then create the heatmap matrix
            zscore_channel_tmas: pd.DataFrame = cmt_df.groupby(by="channel", sort=True).apply(
                lambda group: self._create_r_c_tma_matrix(group, n_cols, n_rows, qc_col)
            )
            zscore_channel_matrices: np.ndarray = np.array(
                [c_tma[0] for c_tma in zscore_channel_tmas.values],
            )

            avg_zscore = np.mean(zscore_channel_matrices, axis=0)

            zscore_channels_matrix.append(avg_zscore)

        return xr.DataArray(
            data=np.stack(zscore_channels_matrix),
            coords=[self.qc_cols, np.arange(n_cols), np.arange(n_rows)],
            dims=["qc_col", "cols", "rows"],
        )


@dataclass
class QCControlMetrics:
    """Computes QC Metrics for a set of control sample FOVs across various runs, and saves the QC
    files in the `longitudinal_control_metrics_dir`.

    Args:
        cohort_path (Union[str,pathlib.Path]): The directory where the extracted images are
        stored for the control FOVs.
        longitudinal_control_metrics_dir (Union[str, pathlib.Path]): The directory where to save
        the QC Control metrics.
        qc_metrics (List[str]): A list of QC metrics to use. Options include:

                - `"Non-zero mean intensity"`
                - `"Total intensity"`
                - `"99.9% intensity value"`

    Attributes:
        qc_cols (List[str]): A list of the QC columns.
        qc_suffixes (List[str]): A list of the QC suffixes, ordered w.r.t `qc_cols`.
    """

    qc_metrics: Optional[List[str]]
    cohort_path: Union[str, pathlib.Path]
    metrics_dir: Union[str, pathlib.Path]

    # Fields initialized after `__post_init__`
    qc_cols: List[str] = field(init=False)
    qc_suffixes: List[str] = field(init=False)
    longitudinal_control_metrics: Dict[Tuple[str, str], pd.DataFrame] = field(init=False)

    def __post_init__(self):
        """Initialize QCControlMetrics."""
        # Input validation: Ensure that the paths exist
        io_utils.validate_paths([self.cohort_path, self.metrics_dir])

        self.qc_cols, self.qc_suffixes = qc_filtering(qc_metrics=self.qc_metrics)

        self.longitudinal_control_metrics = {}

    def compute_control_qc_metrics(
        self,
        control_sample_name: str,
        fovs: List[str],
        channel_exclude: List[str] = None,
        channel_include: List[str] = None,
    ) -> None:
        """Computes QC metrics for a set of Control Sample FOVs and saves their QC files in the
        `longitudinal_control_metrics_dir`. Calculates the following metrics for the specified
        control samples:
                - `"Non-zero mean intensity"`
                - `"Total intensity"`
                - `"99.9% intensity value"`.

        Args:
            control_sample_name (str): An identifier for naming the control sample.
            fovs (List[str]): A list of control samples to find QC metrics for.
            channel_exclude (List[str], optional): A list of channels to exclude. Defaults to None.
            channel_include (List[str], optional): A list of channels to include. Defaults to None.


        Raises:
            ValueError: Errors if `tissues` is either None, or a list of size 0.
        """
        if fovs is None or not isinstance(fovs, list):
            raise ValueError("The tissues must be specified as a list of strings")

        with tqdm(
            total=len(fovs),
            desc=f"Computing QC Longitudinal Control metrics - {control_sample_name}",
            unit="FOVs",
        ) as pbar:
            for fov in ns.natsorted(fovs):
                # Gather the qc files for the current fov if they exist
                pre_computed_metrics = filter(
                    lambda f: "combined" not in f,
                    io_utils.list_files(
                        dir_name=self.metrics_dir,
                        substrs=[f"{fov}_{qc_suffix}.csv" for qc_suffix in self.qc_suffixes],
                    ),
                )

                if len(list(pre_computed_metrics)) != len(self.qc_cols):
                    compute_qc_metrics(
                        extracted_imgs_path=self.cohort_path,
                        fov_name=fov,
                        save_csv=self.metrics_dir,
                    )
                    pbar.set_postfix(FOV=fov, status="Computing")
                else:
                    pbar.set_postfix(FOV=fov, status="Already Computed")
                pbar.update()

        # Combine metrics for the set of FOVs into a single file per QC metric
        for qc_col, qc_suffix in zip(self.qc_cols, self.qc_suffixes):
            metric_files = filter(
                lambda f: "combined" not in f,
                io_utils.list_files(
                    dir_name=self.metrics_dir,
                    substrs=[f"{fov}_{qc_suffix}.csv" for fov in fovs],
                ),
            )

            # Define an aggregated metric DataFrame, and filter channels
            combined_lc_df: pd.DataFrame = _channel_filtering(
                df=pd.concat(
                    (pd.read_csv(os.path.join(self.metrics_dir, mf)) for mf in metric_files),
                ),
                channel_include=channel_include,
                channel_exclude=channel_exclude,
            )

            self.longitudinal_control_metrics.update(
                {(control_sample_name, qc_col): combined_lc_df}
            )

            combined_lc_df.to_csv(
                os.path.join(self.metrics_dir, f"{control_sample_name}_combined_{qc_suffix}.csv"),
                index=False,
            )

    def transformed_control_effects_data(
        self, control_sample_name: str, qc_metric: str, to_csv: bool = False
    ) -> pd.DataFrame:
        """Creates a transformed DataFrame for the Longitudinal Control effects data, normalizing by the mean,
        then taking the `log2` of each value.

        Args:
            control_sample_name (str): A control sample to tranform the longitudinal control effects for.
            qc_metric (str): The metric to transform.
            to_csv (bool, optional): Whether to save the transformed data to a csv. Defaults to False.

        Returns:
            pd.DataFrame: The transformed QC Longitudinal Control data.
        """
        misc_utils.verify_in_list(user_metric=qc_metric, qc_metrics=self.qc_cols)

        try:
            df: pd.DataFrame = self.longitudinal_control_metrics[control_sample_name, qc_metric]
        except KeyError:
            # A qc file which isn't stored in the longitudinal_control_metrics dictionary, try to load it
            # in if it exists as a file
            df: pd.DataFrame = pd.read_csv(
                os.path.join(
                    self.metrics_dir,
                    f"{control_sample_name}_combined_{self.qc_suffixes[self.qc_cols.index(qc_metric)]}.csv",
                )
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"QC Metric Not Found for the Control Sample {control_sample_name}"
            ) from e

        # Apply a log2 transformation to the mean normalized data.
        log2_norm_df: pd.DataFrame = df.pivot(
            index="channel", columns="fov", values=qc_metric
        ).transform(func=lambda row: np.log2(row / row.mean()), axis=1)

        mean_log2_norm_df: pd.DataFrame = (
            log2_norm_df.mean(axis=0)
            .to_frame(name="mean")
            .transpose()
            .sort_values(by="mean", axis=1)
        )

        transformed_df: pd.DataFrame = pd.concat(
            objs=[log2_norm_df, mean_log2_norm_df]
        ).sort_values(by="mean", axis=1, inplace=False)

        transformed_df.rename_axis("channel", axis=0, inplace=True)
        transformed_df.rename_axis("fov", axis=1, inplace=True)

        # Save the pivoted dataframe to a csv
        if to_csv:
            qc_suffix: str = self.qc_suffixes[self.qc_cols.index(qc_metric)]
            transformed_df.to_csv(
                os.path.join(
                    self.metrics_dir,
                    f"{control_sample_name}_transformed_{qc_suffix}.csv",
                ),
                index=True,
            )

        return transformed_df
