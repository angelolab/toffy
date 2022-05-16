import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mibi_bin_tools import bin_files


def compute_mph_metrics(bin_file_path, target, save_csv=True, mass_range=None,):

    # retrieve the total counts and compute pulse heights for each FOV run file
    # saves individual .csv  files to bin_file_path
    total_counts = bin_files.get_total_counts(bin_file_path)
    metric_csvs = {}

    for i in range(1, len(total_counts) + 1):
        pulse_height_file = 'fov-{}-pulse_height.csv'.format(i)

        if mass_range is None:
            median = bin_files.get_median_pulse_height(bin_file_path,
                                                       'fov-{}-scan-1'.format(i), target)
        else:
            median = bin_files.get_median_pulse_height(bin_file_path, 'fov-{}-scan-1'.format(i),
                                                       target, mass_range)
        count = total_counts['fov-{}-scan-1'.format(i)]

        out_df = pd.DataFrame({
            'fov': [i],
            'MPH': [median],
            'total_count': [count]})

        metric_csvs['fov-{}-scan-1'.format(i)] = out_df

        if not os.path.exists(os.path.join(bin_file_path, pulse_height_file)):
            if save_csv:
                out_df.to_csv(os.path.join(bin_file_path, pulse_height_file), index=False)

    if not save_csv:
        return metric_csvs


def combine_mph_metrics(bin_file_path, output_dir):
    total_counts = bin_files.get_total_counts(bin_file_path)
    pulse_heights = []
    fov_counts = []

    for i in range(1, len(total_counts) + 1):
        temp_df = pd.read_csv(os.path.join(bin_file_path, 'fov-{}-pulse_height.csv'.format(i)))
        pulse_heights.append(temp_df['MPH'].values[0])
        if i > 1:
            fov_counts.append(temp_df['total_count'].values[0] + fov_counts[i - 2])
        else:
            fov_counts.append(temp_df['total_count'].values[0])

    combined_df = pd.DataFrame({'pulse_heights': pulse_heights, 'cum_total_count': fov_counts})
    combined_df.to_csv(os.path.join(output_dir, 'total_count_vs_mph_data.csv'), index=False)


def visualize_mph(mph_df, regression: bool, save_dir=None):

    # visualize the median pulse heights
    plt.style.use('dark_background')
    plt.title('FOV total counts vs median pulse height')
    plt.scatter('cum_total_count', 'pulse_heights', data=mph_df)
    plt.gca().set_xlabel('FOV cumulative count')
    plt.gca().set_ylabel('median pulse hight')
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.xlim(0, max(mph_df['cum_total_count']) + 10000)
    if not regression and save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'fov_vs_mph.jpg'))
        return

    if regression:
        # plot with regression line
        x = np.array(mph_df['cum_total_count'])
        y = np.array(mph_df['pulse_heights'])
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m * x + b)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'fov_vs_mph_regression.jpg'))
