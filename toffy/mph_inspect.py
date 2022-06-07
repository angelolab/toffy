import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ark.utils import io_utils
from mibi_bin_tools import bin_files


def bin_array(arr, bin_factor):
    arr_bin = np.cumsum(arr)
    arr_bin[bin_factor:] = arr_bin[bin_factor:] - arr_bin[:-bin_factor]
    arr_bin = arr_bin[bin_factor::bin_factor]

    return arr_bin


def compute_mph_intensities(bin_file_dir, channel, panel, fov_list=None, bin_factor=100):
    """ Compute the median pulse height intensities for given FOVs

    Args:
        bin_file_dir: path to the FOV bin files
        channel: channel to use
        panel: info for bin file extraction
        fov_list: which FOVs to include, if None will include all in data_dir
        bin_factor: size of the bins for the MPH histograms, default 100

    Returns:
        pd.DataFrame containing the MPH intensity data

    """

    # visualize all FOVs in folder if list not specified
    if fov_list is None:
        fov_list = io_utils.remove_file_extensions(io_utils.list_files(bin_file_dir, substrs='.bin'))

    out_df = []

    for fov in fov_list:
        _, intensities, pulse_counts = bin_files.get_histograms_per_tof(bin_file_dir, fov, channel, panel)

        int_bin = np.cumsum(intensities) / intensities.sum()
        median = (np.abs(int_bin - 0.5)).argmin()

        out_df.append({
            'fov': fov,
            'all_intensities': intensities,
            'all_pulse_counts': pulse_counts,
            'median_intensity': median,
        })
    final_df = pd.DataFrame(out_df)

    # create column of binned intensity values and output df to csv
    final_df['binned_intensities'] = final_df['all_intensities'].apply(lambda x: bin_array(x, bin_factor))
    final_df.to_csv(os.path.join(bin_file_dir, 'mph_counts.csv'), index=False)

    return final_df


def visualize_mph_hist(bin_file_dir, channel, panel, fov_list=None, bin_factor=100, x_cutoff=20000, normalize=True):
    """ Create a histogram of the median pulse height intensities for given FOVs

    Args:
        bin_file_dir: path to the FOV bin files
        channel: channel to use
        panel: info for bin file extraction
        fov_list: which FOVs to include, if None will include all in data_dir
        bin_factor: size of the bins for the MPH histograms, default 100
        x_cutoff:
        normalize: whether to normalize the histograms

    """

    data = compute_mph_intensities(bin_file_dir, channel, panel, fov_list, bin_factor)

    # plot each FOV
    plt.style.use('dark_background')
    for idx, row in data.iterrows():
        if normalize:
            plt.plot(np.arange(row['binned_intensities'].shape[0])[0:x_cutoff // bin_factor] * bin_factor,
                     row['binned_intensities'][0:x_cutoff // bin_factor] / np.max(row['binned_intensities']))
        else:
            plt.plot(np.arange(row['binned_intensities'].shape[0])[0:x_cutoff // bin_factor] * bin_factor,
                     row['binned_intensities'][0:x_cutoff // bin_factor])

    plt.gca().set_xlabel('pulse height')
    plt.gca().set_ylabel('occurrence count')
    plt.gcf().set_size_inches(18.5, 10.5)
