import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from natsort import natsort_keygen

from mibi_bin_tools import bin_files
from ark.utils import io_utils
from toffy.normalize import combine_run_metrics


def get_estimated_time(bin_file_dir, fov):
    """Retrieve run time data for each fov json file
    Args:
        bin_file_dir (str): path to the FOV bin and json files
        fov (str): name of fov to get estimated time for
    Returns:
        fov_time (int): estimated run time for the given fov
    """

    # path validation
    io_utils.validate_paths(bin_file_dir)

    # get fov json file in bin_file_path
    json_file = io_utils.list_files(bin_file_dir, fov+".json")
    if len(json_file) == 0:
        raise FileNotFoundError(f"The FOV name supplied doesn't have a JSON file: {fov}")

    # retrieve estimated time (frame dimensions x pixel dwell time)
    with open(os.path.join(bin_file_dir, json_file[0])) as file:
        run_metadata = json.load(file)
        try:
            size = run_metadata.get('frameSize')
            time = run_metadata.get('dwellTimeMillis')
            estimated_time = int(size**2 * time)
        except TypeError:
            raise KeyError("The FOV json file is missing one of the necessary keys "
                           "(frameSize or dwellTimeMillis)")

    return estimated_time


def generate_time_ticks(mph_df):
    """Create a time axis for median pulse heights with ticks at approx. 6 hour increments
    Args:
         mph_df: contains mph date, specifically requires cum_total_count and cum_total_time
         columns
    Returns:
        list of two lists detailing tick locations and tick number labels
    """

    # determine number of ticks and what the labels should be based on total run time
    sub_df = mph_df[['cum_total_count', 'cum_total_time']]
    total_time = sub_df['cum_total_time'].iloc[-1]
    tick_num = int(total_time / (6*(3600*1000)))
    tick_labels = [i * 6 for i in range(0, tick_num+1)]
    time_ticks = [tick*(3600*1000) for tick in tick_labels[1:len(tick_labels)]]

    # find count value associated with the time closest to each tick
    tick_locations = [0]
    for tick in time_ticks:
        count_tick = (sub_df.iloc[(sub_df['cum_total_time']
                                   - tick).abs().argsort()[:1]])['cum_total_count']
        count_tick = (count_tick.to_string()).split(' ')[4]
        tick_locations.append(int(count_tick)/1000000)

    return [tick_locations, tick_labels]


def compute_mph_metrics(bin_file_dir, csv_dir, fov, mass=98, mass_start=97.5, mass_stop=98.5):
    """Retrieves total counts, pulse heights, & estimated time for a given FOV
        Args:
            bin_file_dir (str): path to the FOV bin and json files
            csv_dir (str): path to output csv to
            fov (string): name of fov bin file without the extension
            mass (float): mass for the panel
            mass_start (float): beginning of mass integration range
            mass_stop (float): end of mass integration range
            """

    target = None
    panel = pd.DataFrame([{
        'Mass': mass,
        'Target': target,
        'Start': mass_start,
        'Stop': mass_stop,
    }])

    # retrieve the data from bin file and output to individual csv
    pulse_height_file = fov + '-mph_pulse.csv'

    try:
        median = bin_files.get_median_pulse_height(bin_file_dir, fov,
                                                   target, panel)
        count_dict = bin_files.get_total_counts(bin_file_dir, [fov])
    except FileNotFoundError:
        raise FileNotFoundError(f"The FOV name supplied doesn't have a JSON file: {fov}")

    count = count_dict[fov]
    time = get_estimated_time(bin_file_dir, fov)

    out_df = pd.DataFrame({
        'fov': [fov],
        'MPH': [median],
        'total_count': [count],
        'time': [time]})

    # saves individual .csv files to csv_dir
    out_df.to_csv(os.path.join(csv_dir, pulse_height_file), index=False)


def combine_mph_metrics(csv_dir, return_data=False):
    """Combines data from individual csvs into one
        Args:
            csv_dir (str): path where FOV mph data csvs are stored
            return_data (bool): whether to return dataframe with mph metrics, default False

        Returns:
            combined mph data for all FOVs
            """

    # path validation checks
    io_utils.validate_paths(csv_dir)

    # combine individual csv files
    combine_run_metrics(csv_dir, 'mph_pulse')

    # calculate cumulative sums of total counts and time
    combined_df = pd.read_csv(os.path.join(csv_dir, 'mph_pulse_combined.csv'))
    combined_df = combined_df.sort_values(by="fov", key=natsort_keygen())
    combined_df['cum_total_count'] = combined_df['total_count'].cumsum()
    combined_df['cum_total_time'] = combined_df['time'].cumsum()

    combined_df.to_csv(os.path.join(csv_dir, 'mph_pulse_combined.csv'), index=False)

    # return data
    if return_data:
        return combined_df


def visualize_mph(mph_df, out_dir, regression: bool = False):
    """Create a scatterplot visualizing median pulse heights by FOV cumulative count
        Args:
            mph_df (pd.DataFrame): data detailing total counts and pulse heights
            out_dir (str): path of directory to save plot to
            regression (bool): whether to plot regression line, default is False
            """

    # path validation checks
    if out_dir is not None:
        io_utils.validate_paths(out_dir)

    # visualize the median pulse heights
    plt.style.use('dark_background')
    # plt.title('FOV total counts vs median pulse height')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    x = mph_df['cum_total_count']/1000000
    y = mph_df['MPH']
    ax1.set_xlabel('FOV cumulative count (in millions)')
    ax1.set_ylabel('median pulse height')
    ax1.scatter(x, y)
    ax2 = ax1.twiny()
    ax2.set_xlabel('estimated time (hours)')

    # create time axis
    new_ticks = generate_time_ticks(mph_df)
    tick_locations = new_ticks[0]
    tick_labels = new_ticks[1]
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(tick_locations)
    ax2.set_xticklabels(tick_labels)
    plt.gcf().set_size_inches(18.5, 10.5)

    # plot regression line
    if regression:
        # plot with regression line
        x2 = np.array(mph_df['cum_total_count']/1000000)
        y2 = np.array(mph_df['MPH'])
        m, b = np.polyfit(x2, y2, 1)
        ax1.plot(x2, m * x2 + b)

    # save figure
    file_path = os.path.join(out_dir, 'fov_vs_mph.jpg')
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path)
