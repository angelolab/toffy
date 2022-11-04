import copy
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from requests.exceptions import HTTPError
from scipy.ndimage import gaussian_filter
import seaborn as sns
from shutil import rmtree

from toffy.mibitracker_utils import MibiTrackerError
from toffy.mibitracker_utils import MibiRequests
from toffy import settings
from tmi.image_utils import save_image

import ark.utils.io_utils as io_utils
import ark.utils.misc_utils as misc_utils
import ark.utils.load_utils as load_utils
from mibi_bin_tools import bin_files


# needed to prevent UserWarning: low contrast image barf when saving images
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def create_mibitracker_request_helper(email, password):
    """Create a mibitracker request helper to access a user's MIBItracker info on Ionpath

    Args:
        email (str):
            The user's MIBItracker email address
        password (str):
            The user's MIBItracker password

    Returns:
        ark.mibi.mibitracker_utils.MibiRequests:
            A request helper module instance to access a user's MIBItracker info
    """

    try:
        return MibiRequests(settings.MIBITRACKER_BACKEND, email, password)
    except HTTPError:
        print("Invalid MIBItracker email or password provided")


def download_mibitracker_data(email, password, run_name, run_label, base_dir, tiff_dir,
                              overwrite_tiff_dir=False, img_sub_folder=None,
                              fovs=None, channels=None):
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
    if len(run_info['results']) == 0:
        raise ValueError('No data found for run_name %s and run_label %s' % (run_name, run_label))

    # extract the name of the FOVs and their associated internal IDs
    run_fov_names = [img['number'] for img in run_info['results'][0]['imageset']['images']]
    run_fov_ids = [img['id'] for img in run_info['results'][0]['imageset']['images']]

    # if fovs is None, ensure all of the fovs in run_fov_names_ids are chosen
    if fovs is None:
        fovs = run_fov_names

    # ensure all of the fovs are valid (important if the user explicitly specifies fovs)
    misc_utils.verify_in_list(
        provided_fovs=fovs,
        mibitracker_run_fovs=run_fov_names
    )

    # iterate through each fov and get the longest set of channels
    # this is to prevent getting no channels if the first FOV is a Moly point or is incomplete
    image_data = run_info['results'][0]['imageset']['images']
    _, run_channels = max(
        {i: image_data[i]['pngs'] for i in np.arange(len(image_data))}.items(),
        key=lambda x: len(set(x[1]))
    )

    # if channels is None, ensure all of the channels in run_channels are chosen
    if channels is None:
        channels = run_channels

    # ensure all of the channels are valid (important if the user explicitly specifies channels)
    misc_utils.verify_in_list(
        provided_chans=channels,
        mibitracker_run_chans=run_channels
    )

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
    for img in run_info['results'][0]['imageset']['images']:
        # if the image fov name is not specified, move on
        if img['number'] not in fovs:
            continue

        print("Creating data for fov %s" % img['number'])

        # make the fov directory
        os.mkdir(os.path.join(base_dir, tiff_dir, img['number']))

        # make the img_sub_folder inside the fov directory if specified
        if len(img_sub_folder) > 0:
            os.mkdir(os.path.join(base_dir, tiff_dir, img['number'], img_sub_folder))

        # iterate over each provided channel
        for chan in channels:
            # extract the channel data from MIBItracker as a numpy array
            # fail on the whole FOV if the channel is not found (most likely a Moly point)
            try:
                chan_data = mr.get_channel_data(img['id'], chan)
            except MibiTrackerError as mte:
                print("On FOV %s, failed to download channel %s, moving on to the next FOV. "
                      "If FOV %s is a Moly point, ignore. "
                      "Otherwise, please ensure that channel %s exists on MibiTracker for FOV %s"
                      % (img['number'], chan, img['number'], chan, img['number']))

                # clean the FOV: we will not have a folder for it (in case of Moly point)
                rmtree(os.path.join(base_dir, tiff_dir, img['number']))

                # do not attempt to download any more channels
                break

            # define the name of the channel file
            chan_file = '%s.tiff' % chan

            # write the data to a .tiff file in the FOV directory structure
            fname: str = os.path.join(base_dir, tiff_dir, img['number'], img_sub_folder, chan_file)
            save_image(fname, chan_data)

        # append the run name and run id to the list
        run_order.append((img['number'], img['id']))

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
        suffix_ignore = ''

    # TODO: if anyone can do this using a walrus operator I'd appreciate it!
    return sorted(
        fovs,
        key=lambda f: (
            int(f.replace(suffix_ignore, '').split('-')[1]),
            int(f.replace(suffix_ignore, '').split('-')[3])
        )
    )


def compute_qc_metrics(bin_file_path, extracted_imgs_path, fov_name,
                       gaussian_blur=False, blur_factor=1, save_csv=None):
    """Compute the QC metric matrices for the image data provided

    Args:
        bin_file_path (str):
            the directory to the MIBI bin files for extraction,
            also where the fov-level QC metric files will be written
        extracted_imgs_path (str):
            the directory when extracted images are stored
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
        None | Dict[str, pd.DataFrame]:
            If save_csv is False, returns qc metrics. Otherwise, no return
    """

    # path validation checks
    if not os.path.exists(bin_file_path):
        raise FileNotFoundError("bin_file_path %s does not exist" % bin_file_path)

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

    Returns:
        Dict[str, pd.DataFrame]:
            Returns qc metrics

    """

    # there's only 1 FOV and 1 type ('pulse'), so subset on that
    image_data = image_data.loc[fov_name, 'pulse', :, :, :]

    # define the list of channels to use
    chans = image_data.channel.values

    # define numpy arrays for all the metrics to extract, more efficient indexing than pandas
    blank_arr = np.zeros(image_data.shape[2], dtype='float32')
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
                image_data_np, sigma=blur_factor, mode='nearest', truncate=2.0
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
        metric_df = pd.DataFrame(
            columns=['fov', 'channel', mc], dtype=object
        )

        # assign the metric data
        metric_df[mc] = md

        # assign the fov and channel names
        metric_df['fov'] = fov_name
        metric_df['channel'] = chans

        metric_csvs[f'{fov_name}_{ms}.csv'] = metric_df

    return metric_csvs


def combine_qc_metrics(qc_metrics_dir):
    """Aggregates the QC results of each FOV into one `.csv`

    Args:
        qc_metrics_dir (str):
            the name of the folder containing the QC metric files
    """

    # path validation check
    if not os.path.exists(qc_metrics_dir):
        raise FileNotFoundError('qc_metrics_dir %s does not exist' % qc_metrics_dir)

    for ms in settings.QC_SUFFIXES:
        # define an aggregated metric DataFrame
        metric_df = pd.DataFrame()

        # list all the files corresponding to this metric
        metric_files = io_utils.list_files(qc_metrics_dir, substrs=ms + '.csv')

        # don't consider any existing combined .csv files, just the fov-level .csv files
        metric_files = [
            mf for mf in metric_files if 'combined' not in mf
        ]

        # sort the files to ensure consistency
        metric_files = sort_bin_file_fovs(metric_files, suffix_ignore='_%s.csv' % ms)

        # iterate over each metric file and append the data to metric_df
        for mf in metric_files:
            metric_df = pd.concat([metric_df, pd.read_csv(os.path.join(qc_metrics_dir, mf))])

        # write the aggregated metric data
        # NOTE: if this combined metric file already exists, it will be overwritten
        metric_df.to_csv(os.path.join(qc_metrics_dir, 'combined_%s.csv' % ms), index=False)


def visualize_qc_metrics(metric_name, qc_metric_dir, axes_size=16, wrap=6,
                         dpi=None, save_dir=None, ax=None):
    """Visualize a barplot of a specific QC metric

    Args:
        metric_name (str):
            The name of the QC metric, used as the y-axis label
        qc_metric_dir (str):
            The path to the directory containing the `'combined_{qc_metric}.csv'` files
        axes_size (int):
            The font size of the axes labels
        wrap (int):
            How many plots to display per row
        dpi (int):
            If saving, the resolution of the image to use
            Ignored if save_dir is None
        save_dir (str):
            If saving, the name of the directory to save visualization to
        ax (matplotlib.axes.Axes):
            Axes to place catplots
    """

    # verify the metric provided is valid
    if metric_name not in settings.QC_COLUMNS:
        raise ValueError("Invalid metric %s provided, must be set to 'Non-zero mean intensity', "
                         "'Total intensity', or '99.9%% intensity value'" % metric_name)

    # verify the path to the QC metric datasets exist
    if not os.path.exists(qc_metric_dir):
        raise FileNotFoundError('qc_metric_dir %s does not exist' % qc_metric_dir)

    # get the file name of the combined QC metric .csv file to use
    qc_metric_index = settings.QC_COLUMNS.index(metric_name)
    qc_metric_suffix = settings.QC_SUFFIXES[qc_metric_index]
    qc_metric_path = os.path.join(qc_metric_dir, 'combined_%s.csv' % qc_metric_suffix)

    # ensure the user set the right qc_metric_dir
    if not os.path.exists(qc_metric_path):
        raise FileNotFoundError('Could not locate %s, ensure qc_metric_dir is correct' %
                                qc_metric_path)

    # read in the QC metric data
    qc_metric_df = pd.read_csv(qc_metric_path)

    # filter out naturally-occurring elements as well as Noodle
    qc_metric_df = qc_metric_df[~qc_metric_df['channel'].isin(settings.QC_CHANNEL_IGNORE)]

    # filter out anything prefixed with 'chan_'
    qc_metric_df = qc_metric_df[~qc_metric_df['channel'].str.startswith('chan_')]

    # catplot allows for easy facets on a barplot
    g = sns.catplot(
        x='fov',
        y=metric_name,
        col='channel',
        col_wrap=wrap,
        data=qc_metric_df,
        kind='bar',
        color='black',
        sharex=True,
        sharey=False,
        ax=ax,
    )

    # per Erin's visualization, don't show the hundreds of fov labels on the x-axis
    _ = plt.xticks([])

    # remove the 'channel =' in each subplot title
    _ = g.set_titles(template='{col_name}')

    # per Erin's visualization remove the default axis title on the y-axis
    # and instead show 'fov' along x-axis and the metric name along the y-axis (overarching)
    _ = g.set_axis_labels('', '')
    _ = g.fig.text(
        x=0.5,
        y=0,
        horizontalalignment='center',
        s='fov',
        size=axes_size
    )
    _ = g.fig.text(
        x=0,
        y=0.5,
        verticalalignment='center',
        s=metric_name,
        size=axes_size,
        rotation=90
    )

    # save the figure if specified
    if save_dir:
        misc_utils.save_figure(save_dir, '%s_barplot_stats.png' % metric_name, dpi=dpi)


def format_img_data(img_data):
    """ Formats the image array from load_imgs_from_tree to be same structure as the array returned
    by extract_bin_files. Works for one FOV data at a time.
    Args:
        img_data (str): current image data array as produced by load function
    Returns:
         xarray.DataArray: image data array with shape [fov, type, x, y, channel]
    """

    # add type dimension
    img_data = img_data.assign_coords(type='pulse')
    img_data = img_data.expand_dims('type', 1)

    # edit dimension names
    img_data = img_data.rename({'fovs': 'fov', 'rows': 'x', 'cols': 'y', 'channels': 'channel'})

    return img_data
