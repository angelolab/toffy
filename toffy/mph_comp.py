import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mibi_bin_tools import bin_files


def compute_mph_metrics(bin_file_path, target, save_csv=True, mass_range=(-0.3, 0.0)):
    """Retrieves FOV total counts and median pulse heights for all bin files in the directory
        Args:
            bin_file_path (str): path to the FOV bin and json files
            target (str): channel to use
            save_csv (bool): whether to save to csv file or output data, defaults to True
            mass_range (tuple): integration range for pulse heights, default same as pulse function

        Return:
            None | Dict[str, pd.DataFrame]: if save_csv if False, return mph metrics
            """

    # get total counts for each FOV
    total_counts = bin_files.get_total_counts(bin_file_path)
    metric_csvs = {}

    # retrieve the data from each bin file and store it / output to individual csv
    for i in range(1, len(total_counts) + 1):
        pulse_height_file = 'fov-{}-pulse_height.csv'.format(i)

        # get median pulse heights
        median = bin_files.get_median_pulse_height(bin_file_path, 'fov-{}-scan-1'.format(i),
                                                   target, mass_range)
        count = total_counts['fov-{}-scan-1'.format(i)]

        out_df = pd.DataFrame({
            'fov': [i],
            'MPH': [median],
            'total_count': [count]})

        metric_csvs['fov-{}-scan-1'.format(i)] = out_df

        # saves individual .csv  files to bin_file_path
        if not os.path.exists(os.path.join(bin_file_path, pulse_height_file)):
            if save_csv:
                out_df.to_csv(os.path.join(bin_file_path, pulse_height_file), index=False)

    # return data
    if not save_csv:
        return metric_csvs


def combine_mph_metrics(bin_file_path, output_dir):
    """Combines data from individual csvs into one
        Args:
            bin_file_path (str): path to the FOV bin and json files
            output_dir (str): path to output csv to
            """

    # get FOV total counts
    total_counts = bin_files.get_total_counts(bin_file_path)
    pulse_heights = []
    fov_counts = []

    # for each csv retrieve mph values
    for i in range(1, len(total_counts) + 1):
        temp_df = pd.read_csv(os.path.join(bin_file_path, 'fov-{}-pulse_height.csv'.format(i)))
        pulse_heights.append(temp_df['MPH'].values[0])
        # calculate total counts cumulatively for plotting
        if i > 1:
            fov_counts.append(temp_df['total_count'].values[0] + fov_counts[i - 2])
        else:
            fov_counts.append(temp_df['total_count'].values[0])

    # save csv to output_dir
    combined_df = pd.DataFrame({'pulse_heights': pulse_heights, 'cum_total_count': fov_counts})
    combined_df.to_csv(os.path.join(output_dir, 'total_count_vs_mph_data.csv'), index=False)


def visualize_mph(mph_df, regression: bool, save_dir=None):
    """Create a scatterplot visualizing median pulse heights by FOV cumulative count
        Args:
            mph_df (pd.DataFrame): data detailing total counts and pulse heights
            regression (bool): whether or not to plot regression line
            save_dir (str): path of directory to save plot to
            """

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
