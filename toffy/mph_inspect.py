import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mibi_bin_tools.bin_files import get_histograms_per_tof
from tmi import io_utils, misc_utils


def bin_array(arr, bin_factor):
    """ Bin data for visualization according to a bin_factor
    Args:
        arr (array_like): original data
        bin_factor (int): size of the bins for the histograms

    Returns:
        the binned data

    """
    arr_bin = np.cumsum(arr)
    arr_bin[bin_factor:] = arr_bin[bin_factor:] - arr_bin[:-bin_factor]
    arr_bin = arr_bin[bin_factor::bin_factor]

    return arr_bin


def compute_intensities(bin_file_dir, fov_list, mass, mass_start, mass_stop, bin_factor=100):
    """ Compute the pulse height intensities for given FOVs

    Args:
        bin_file_dir (str): path to the FOV bin files
        fov_list (list): which FOVs to include
        mass (float): mass for the panel
        mass_start (float): beginning of mass integration range
        mass_stop (float): end of mass integration range
        bin_factor (int): size of the bins for the histograms, default 100

    Returns:
        pd.DataFrame containing the MPH intensity data

    """

    panel = pd.DataFrame([{
        'Mass': mass,
        'Target': None,
        'Start': mass_start,
        'Stop': mass_stop,
    }])

    out_df = []

    # retrieve and store data for each fov
    for fov in fov_list:
        _, intensities, pulse_counts = get_histograms_per_tof(
            bin_file_dir, fov, None, panel)

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
    final_df['binned_intensities'] = final_df['all_intensities'].apply(lambda x:
                                                                       bin_array(x, bin_factor))
    return final_df


def visualize_intensity_data(bin_file_dir, mass, mass_start, mass_stop, fov_list=None,
                             bin_factor=100, x_cutoff=20000, normalize=True):
    """ Create a histogram of the pulse height intensities for given FOVs

    Args:
        bin_file_dir (str): path to the FOV bin files
        mass (float): mass for the panel
        mass_start (float): beginning of mass integration range
        mass_stop (float): end of mass integration range
        fov_list (list): which FOVs to include, if None will include all in data_dir
        bin_factor (int): size of the bins for the histograms, default 100
        normalize (bool): whether to normalize the histograms

    """

    # validate path
    io_utils.validate_paths(bin_file_dir)

    # compute for all FOVs in folder if list not specified, verify valid fov names
    all_fovs = io_utils.remove_file_extensions(io_utils.list_files
                                               (bin_file_dir, substrs='.bin'))
    if fov_list is None:
        fov_list = all_fovs
    else:
        misc_utils.verify_in_list(provided_fovs=fov_list, fovs_in_directory=all_fovs)

    # compute the pulse height intensities
    data = compute_intensities(bin_file_dir, fov_list, mass, mass_start, mass_stop, bin_factor)

    # plot each FOV
    plt.style.use('dark_background')
    for idx, row in data.iterrows():
        if normalize:
            plt.plot(np.arange(row['binned_intensities'].shape[0])[0:x_cutoff // bin_factor]
                     * bin_factor, row['binned_intensities'][0:x_cutoff // bin_factor]
                     / np.max(row['binned_intensities']))
        else:
            plt.plot(np.arange(row['binned_intensities'].shape[0])[0:x_cutoff // bin_factor]
                     * bin_factor, row['binned_intensities'][0:x_cutoff // bin_factor])

    plt.gca().set_xlabel('pulse height')
    plt.gca().set_ylabel('occurrence count')
    plt.gcf().set_size_inches(18.5, 10.5)
