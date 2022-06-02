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


def compute_mph_intensities(data_dir, channel, panel, fov_list=None, bin_factor=100):
    # if os.path.exists(os.path.join(data_dir, 'mph_counts.csv')):
    #    return

    # visualize all fovs in folder if list not specified
    if fov_list is None:
        fov_list = io_utils.remove_file_extensions(io_utils.list_files(data_dir, substrs='.bin'))

    out_df = []

    for fov in fov_list:
        _, intensities, pulse_counts = bin_files.get_histograms_per_tof(data_dir, fov, channel, panel)

        int_bin = np.cumsum(intensities) / intensities.sum()
        median = (np.abs(int_bin - 0.5)).argmin()

        out_df.append({
            'fov': fov,
            'all_intensities': intensities,
            'all_pulse_counts': pulse_counts,
            'median_intensity': median,
        })

    final_df = pd.DataFrame(out_df)
    final_df['binned_intensities'] = final_df['all_intensities'].apply(lambda x: bin_array(x, bin_factor))
    final_df.to_csv(os.path.join(data_dir, 'mph_counts.csv'))

    return final_df

'''
channel1 = 'Mo98'
panel1 = pd.DataFrame([{
    'Mass': int(channel1[2:]),
    'Target': channel1,
    'Start': float(channel1[2:]) - 0.5,
    'Stop': float(channel1[2:]) + 0.5,
}])
data_dir = os.path.join('data', 'tissue')
data1 = compute_mph_intensities(data_dir, channel1, panel1)
'''


def visualize_mph_hist(base_dir, channel, panel, fov_list=None, bin_factor=100, x_cutoff=20000, normalize=True):
    data = compute_mph_intensities(base_dir, channel, panel, fov_list, bin_factor)
    #data = pd.read_csv(os.path.join(base_dir, 'mph_counts.csv'))
    # matplotlib fun
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

# visualize_mph_hist(data_dir, channel1, panel1)
