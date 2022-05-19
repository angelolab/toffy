import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mibi_bin_tools import bin_files
from ark.utils import io_utils


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

    # path validation checks
    io_utils.validate_paths(bin_file_path)

    # retrieve the data from bin file and store it output to individual csv
    pulse_height_file = fov + '-pulse_height.csv'

    # get median pulse heights
    median = bin_files.get_median_pulse_height(bin_file_path, fov,
                                               target, (mass_start, mass_stop))
    count_dict = bin_files.get_total_counts(bin_file_path, [fov])
    count = count_dict[fov]

    out_df = pd.DataFrame({
        'fov': [fov],
        'MPH': [median],
        'total_count': [count]})

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

    # for each csv retrieve mph values
    for i, file in enumerate(fov_bins):
        temp_df = pd.read_csv(os.path.join(bin_file_path, file + '-pulse_height.csv'))
        pulse_heights.append(temp_df['MPH'].values[0])
        fov_counts.append(temp_df['total_count'].values[0])

    # calculate cumulative sums of total counts
    fov_counts_cum = [fov_counts[j]+fov_counts[j-1] if j > 0 else fov_counts[j]
                      for j in range(len(fov_counts))]

    # save csv to output_dir
    combined_df = pd.DataFrame({'pulse_heights': pulse_heights, 'cum_total_count': fov_counts_cum})
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
    plt.title('FOV total counts vs median pulse height')
    plt.scatter('cum_total_count', 'pulse_heights', data=mph_df)
    plt.gca().set_xlabel('FOV cumulative count')
    plt.gca().set_ylabel('median pulse hight')
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.xlim(0, max(mph_df['cum_total_count']) + 10000)

    # save figure without regression line
    if not regression and save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'fov_vs_mph.jpg'))
        return

    # plot regression line
    if regression:
        x = np.array(mph_df['cum_total_count'])
        y = np.array(mph_df['pulse_heights'])
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m * x + b)
        # save figure
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'fov_vs_mph_regression.jpg'))
