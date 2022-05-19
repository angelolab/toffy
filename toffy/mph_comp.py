import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from mibi_bin_tools import bin_files
from ark.utils import io_utils


def get_estimated_time(bin_file_path):
    """Retrieve run time data for each bin file
    Args:
        bin_file_path (str): path to the FOV bin and json files
    Returns:
        fov_times (dictionary): fov bin file names and estimated run time
    """

    # get json files in bin_file_path
    fov_files = bin_files._find_bin_files(bin_file_path)
    json_files = \
        [(name, os.path.join(bin_file_path, fov['json'])) for name, fov in fov_files.items()]
    fov_times = {}

    # retrieve estimated time (frame dimensions x pixel dwell time)
    for j in json_files:
        with open(j[1]) as file:
            run_metadata = json.load(file)
            size = run_metadata.get('frameSize')
            time = run_metadata.get('dwellTimeMillis')
            estimated_time = int(size**2 * time)
            fov_times[j[0]] = estimated_time

    return fov_times


def compute_mph_metrics(bin_file_path, fov, target, mass_start, mass_stop, save_csv=True):
    """Retrieves FOV total counts and median pulse heights for all bin files in the directory
        Args:
            bin_file_path (str): path to the FOV bin and json files
            fov (string): name of fov bin file without the extension
            target (str): channel to use
            mass_start (float): beginning of mass integration range
            mass_stop (float): end of mass integration range
            save_csv (bool): whether to save to csv file or output data, defaults to True

            """
    # retrieve the total counts and compute pulse heights for each FOV run file
    # saves individual .csv  files to bin_file_path
    total_counts = bin_files.get_total_counts(bin_file_path)
    fov_times = get_estimated_time(bin_file_path)
    fov_keys = list(fov_times.keys())
    metric_csvs = {}

    # path validation checks
    io_utils.validate_paths(bin_file_path)

    # retrieve the data from bin file and store it output to individual csv
    pulse_height_file = fov +'-pulse_height.csv'

    # get median pulse heights
    median = bin_files.get_median_pulse_height(bin_file_path, fov,
                                               target, (mass_start, mass_stop))
    count_dict = bin_files.get_total_counts(bin_file_path, [fov])
    count = count_dict[fov]

    out_df = pd.DataFrame({
        'fov': [fov],
        'MPH': [median],
        'total_count': [count],
        'time': [fov_times[fov_keys[i - 1]]]})

    # saves individual .csv  files to bin_file_path
    if not os.path.exists(os.path.join(bin_file_path, pulse_height_file)):
        if save_csv:
            out_df.to_csv(os.path.join(bin_file_path, pulse_height_file), index=False)


def combine_mph_metrics(bin_file_path, output_dir):
    """Combines data from individual csvs into one
        Args:
            bin_file_path (str): path to the FOV bin and json files
            output_dir (str): path to output csv to
            """

    # path validation checks
    io_utils.validate_paths(bin_file_path)
    io_utils.validate_paths(output_dir)

    # list bin files in directory
    fov_bins = io_utils.list_files(bin_file_path, ".bin")
    fov_bins = io_utils.remove_file_extensions(fov_bins)

    pulse_heights = []
    fov_counts = []
    estimated_time = []

    # for each csv retrieve mph values
    for i, file in enumerate(fov_bins):
        temp_df = pd.read_csv(os.path.join(bin_file_path, file + '-pulse_height.csv'))
        pulse_heights.append(temp_df['MPH'].values[0])
        fov_counts.append(temp_df['total_count'].values[0])
        estimated_time.append(temp_df['time'].values[0])

    # calculate cumulative sums of total counts
    fov_counts_cum = [fov_counts[j]+fov_counts[j-1] if j > 0 else fov_counts[j]
                      for j in range(len(fov_counts))]

    # save csv to output_dir
    combined_df = pd.DataFrame({'pulse_heights': pulse_heights, 'cum_total_count': fov_counts_cum
                               'cum_total_time': estimated_time})
    combined_df.to_csv(os.path.join(output_dir, 'total_count_vs_mph_data.csv'), index=False)


def visualize_mph(mph_df, regression: bool, save_dir=None):
    """Create a scatterplot visualizing median pulse heights by FOV cumulative count
        Args:
            mph_df (pd.DataFrame): data detailing total counts and pulse heights
            regression (bool): whether or not to plot regression line
            save_dir (str): path of directory to save plot to
            """

    # path validation checks
    if save_dir is not None:
        io_utils.validate_paths(save_dir)

    # visualize the median pulse heights
    plt.style.use('dark_background')
    # plt.title('FOV total counts vs median pulse height')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    x = mph_df['cum_total_count']
    y = mph_df['pulse_heights']
    ax1.scatter(x, y)
    ax1.set_xlabel('FOV cumulative count')
    ax1.set_ylabel('median pulse height')
    ax2.set_xlabel('estimated time (ms)')
    ax1.set_xlim(0, max(x) + 10000)
    ax2.set_xlim(0, max(x) + 10000)
    ax2.set_xticks(x)
    ax2.set_xticklabels(mph_df['cum_total_time'])
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.xlim(0, max(mph_df['cum_total_count']) + 10000)

    # save figure without regression line
    if not regression and save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'fov_vs_mph.jpg'))
        return

    # plot regression line
    if regression:
        # plot with regression line
        x2 = np.array(mph_df['cum_total_count'])
        y2 = np.array(mph_df['pulse_heights'])
        m, b = np.polyfit(x2, y2, 1)
        plt.plot(x2, m * x2 + b)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'fov_vs_mph_regression.jpg'))
    plt.show()
