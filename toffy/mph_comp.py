import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from mibi_bin_tools import bin_files
from ark.utils import io_utils


def get_estimated_time(bin_file_path, fov):
    """Retrieve run time data for each fov json file
    Args:
        bin_file_path (str): path to the FOV bin and json files
        fov (str): name of fov to get estimated time for
    Returns:
        fov_time (int): estimated run time for the given fov
    """

    # path validation
    io_utils.validate_paths(bin_file_path)

    # get fov json file in bin_file_path
    json_file = io_utils.list_files(bin_file_path, fov+".json")
    if len(json_file) == 0:
        raise FileNotFoundError(f"The FOV name supplied doesn't have a JSON file: {fov}")

    # retrieve estimated time (frame dimensions x pixel dwell time)
    with open(os.path.join(bin_file_path, json_file[0])) as file:
        run_metadata = json.load(file)
        try:
            size = run_metadata.get('frameSize')
            time = run_metadata.get('dwellTimeMillis')
            estimated_time = int(size**2 * time)
        except TypeError:
            raise KeyError("The FOV json file is missing one of the necessary keys "
                           "(frameSize or dwellTimeMillis)")

    return estimated_time


def compute_mph_metrics(bin_file_path, fov, target, mass_start, mass_stop, save_csv=True):
    """Retrieves FOV total counts, pulse heights, & estimated time for all fovs in the directory
        Args:
            bin_file_path (str): path to the FOV bin and json files
            fov (string): name of fov bin file without the extension
            target (str): channel to use
            mass_start (float): beginning of mass integration range
            mass_stop (float): end of mass integration range
            save_csv (bool): whether to save to csv file or output data, defaults to True

            """

    # path validation
    io_utils.validate_paths(bin_file_path)

    # retrieve the data from bin file and output to individual csv
    pulse_height_file = fov + '-pulse_height.csv'

    try:
        median = bin_files.get_median_pulse_height(bin_file_path, fov,
                                                   target, (mass_start, mass_stop))
        count_dict = bin_files.get_total_counts(bin_file_path, [fov])
    except FileNotFoundError:
        raise FileNotFoundError(f"The FOV name supplied doesn't have a JSON file: {fov}")
    except ValueError:
        raise ValueError(f"The target name is invalid: {target}")

    count = count_dict[fov]
    time = get_estimated_time(bin_file_path, fov)

    out_df = pd.DataFrame({
        'fov': [fov],
        'MPH': [median],
        'total_count': [count],
        'time': [time]})

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
    estimated_time_cum = [estimated_time[j] + estimated_time[j - 1] if j > 0
                          else estimated_time[j] for j in range(len(estimated_time))]

    # save csv to output_dir
    combined_df = pd.DataFrame({'pulse_heights': pulse_heights, 'cum_total_count': fov_counts_cum,
                               'cum_total_time': estimated_time_cum})
    file_path = os.path.join(output_dir, 'total_count_vs_mph_data.csv')
    if os.path.exists(file_path):
        os.remove(file_path)
    combined_df.to_csv(file_path, index=False)


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
    x_alt = mph_df['cum_total_time']
    ax1.scatter(x, y)
    ax1.set_xlabel('FOV cumulative count')
    ax1.set_ylabel('median pulse height')
    ax2.set_xlabel('estimated time (ms)')
    ax2.scatter(x_alt, y)
    plt.gcf().set_size_inches(18.5, 10.5)

    # plot regression line
    if regression:
        # plot with regression line
        x2 = np.array(mph_df['cum_total_count'])
        y2 = np.array(mph_df['pulse_heights'])
        m, b = np.polyfit(x2, y2, 1)
        ax1.plot(x2, m * x2 + b)

    # save figure
    if save_dir is not None:
        file_path = os.path.join(save_dir, 'fov_vs_mph.jpg')
        if os.path.exists(file_path):
            os.remove(file_path)
        plt.savefig(file_path)

    plt.show()
