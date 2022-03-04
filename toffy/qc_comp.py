import copy
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import re
from requests.exceptions import HTTPError
from scipy.ndimage import gaussian_filter
import seaborn as sns
from shutil import rmtree
from skimage.io import imsave

from toffy.mibitracker_utils import MibiTrackerError
from toffy.mibitracker_utils import MibiRequests

import ark.settings as settings
import ark.utils.io_utils as io_utils
import ark.utils.load_utils as load_utils
import ark.utils.misc_utils as misc_utils
import mibi_bin_tools.bin_files as bin_files

# needed to prevent UserWarning: low contrast image barf when saving images
import warnings
warnings.filterwarnings('ignore')


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
            imsave(
                os.path.join(base_dir, tiff_dir, img['number'], img_sub_folder, chan_file),
                chan_data
            )

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


def compute_qc_metrics(bin_file_path, fov_name, panel_path,
                       gaussian_blur=False, blur_factor=1):
    """Compute the QC metric matrices for the image data provided

    Args:
        bin_file_path (str):
            the directory to the MIBI bin files for extraction,
            also where the fov-level QC metric files will be written
        fov_name (str):
            the name of the FOV to extract from `bin_file_path`, needs to correspond with JSON name
        panel_path (str):
            the path to the file defining the panel info for bin file extraction
        gaussian_blur (bool):
            whether or not to add Gaussian blurring
        blur_factor (int):
            the sigma (standard deviation) to use for Gaussian blurring
            set to 0 to use raw inputs without Gaussian blurring
            ignored if `gaussian_blur` set to `False`
    """

    # path validation checks
    if not os.path.exists(bin_file_path):
        raise FileNotFoundError("bin_file_path %s does not exist" % bin_file_path)

    if not os.path.exists(panel_path):
        raise FileNotFoundError("panel_path %s does not exist" % panel_path)

    if not os.path.exists(os.path.join(bin_file_path, fov_name + '.json')):
        raise FileNotFoundError("fov file %s.json not found in bin_file_path" % fov_name)

    # run the bin file extraction, we'll extract all FOVs
    # the image coords should be: ['fov', 'type', 'x', 'y', 'channel']
    image_data = bin_files.extract_bin_files(
        data_dir=bin_file_path,
        out_dir=None,
        include_fovs=[fov_name],
        panel=pd.read_csv(panel_path)
    )

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

    # define a pandas DataFrame for each metric
    df_nonzero_mean = pd.DataFrame(
        columns=['fov', 'channel', 'Non-zero mean intensity'], dtype=object
    )
    df_total_intensity = pd.DataFrame(
        columns=['fov', 'channel', 'Total intensity'], dtype=object
    )
    df_99_9_intensity = pd.DataFrame(
        columns=['fov', 'channel', '99.9 intensity value'], dtype=object
    )

    # append the stats
    df_nonzero_mean['Non-zero mean intensity'] = nonzero_mean_intensity
    df_total_intensity['Total intensity'] = total_intensity
    df_99_9_intensity['99.9 intensity value'] = intensity_99_9

    # append the FOV name and channel name
    df_nonzero_mean['fov'] = fov_name
    df_total_intensity['fov'] = fov_name
    df_99_9_intensity['fov'] = fov_name

    df_nonzero_mean['channel'] = chans
    df_total_intensity['channel'] = chans
    df_99_9_intensity['channel'] = chans

    # save the data to the qc_dir
    df_nonzero_mean.to_csv(
        os.path.join(bin_file_path, '%s_nonzero_mean_stats.csv') % fov_name, index=False
    )
    df_total_intensity.to_csv(
        os.path.join(bin_file_path, '%s_total_intensity_stats.csv') % fov_name, index=False
    )
    df_99_9_intensity.to_csv(
        os.path.join(bin_file_path, '%s_percentile_99_9_stats.csv') % fov_name, index=False
    )


def combine_qc_metrics(bin_file_path):
    """Aggregates the QC results of each FOV into one `.csv`

    Args:
        bin_file_path (str):
            the name of the folder containing the QC metric files
    """

    # path validation check
    if not os.path.exists(bin_file_path):
        raise FileNotFoundError('bin_file_path %s does not exist' % bin_file_path)

    # iterate over each metric substr
    metric_substrs = [
        'nonzero_mean_stats.csv', 'total_intensity_stats.csv', 'percentile_99_9_stats.csv'
    ]
    for ms in metric_substrs:
        # define an aggregated metric DataFrame
        metric_df = pd.DataFrame()

        # list all the files corresponding to this metric
        metric_files = io_utils.list_files(bin_file_path, substrs=ms)

        # this prevents an already-existing combined .csv file from sneaking in
        metric_files = [
            mf_match[0] + '_%s' % ms for mf in metric_files if
            len(mf_match := re.findall(r'fov-\d+-scan-\d+', mf)) == 1
        ]  # go walrus operators

        # sort the files to ensure consistency
        # NOTE: all FOVs are named fov-m-scan-n, 'm' determines run acquisition order
        # TODO: need an additional sort by scan?
        metric_files = sorted(metric_files, key=lambda x: re.findall(r'\d+', x)[0])

        # iterate over each metric file and append the data to metric_df
        for mf in metric_files:
            metric_df = pd.concat([metric_df, pd.read_csv(os.path.join(bin_file_path, mf))])

        # write the aggregated metric data
        # NOTE: if this combined metric file already exists, it will be overwritten
        metric_df.to_csv(os.path.join(bin_file_path, 'combined_%s' % ms), index=False)


def visualize_qc_metrics(qc_metric_df, metric_name, axes_size=16, wrap=6, dpi=None, save_dir=None):
    """Visualize a barplot of a specific QC metric

    Args:
        qc_metric_df (pandas.DataFrame):
            A QC metric matrix as returned by `compute_qc_metrics`, melted
        metric_name (str):
            The name of the QC metric, used as the y-axis label
        axes_size (int):
            The font size of the axes labels
        wrap (int):
            How many plots to display per row
        dpi (int):
            If saving, the resolution of the image to use
            Ignored if save_dir is None
        save_dir (str):
            If saving, the name of the directory to save visualization to
    """

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
        sharey=False
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
    if save_dir is not None:
        misc_utils.save_figure(save_dir, '%s_barplot_stats.png' % metric_name, dpi=dpi)
