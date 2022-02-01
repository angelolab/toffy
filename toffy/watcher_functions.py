import numpy as np
import pandas as pd
import os

from mibi_bin_tools.bin_files import extract_bin_files, get_histograms_per_tof


def write_counts_per_mass(base_dir, fov, masses, integration_window=(-0.5, 0.5)):
    """Records the total counts per mass for the specified FOV

    Args:
        base_dir (str): the directory containing the FOV
        fov (str): the name of the fov to extract
        masses (list): the list of masses to extract counts from
        integration_window (tuple): start and stop offset for integrating mass peak
    """

    start_offset, stop_offset = integration_window

    # create panel with the extraction details for each mass
    mass_starts = [mass + start_offset for mass in masses]
    mass_stops = [mass + stop_offset for mass in masses]
    targets = ['placeholder' for _ in masses]

    panel = pd.DataFrame({
        'Mass': masses,
        'Target': targets,
        'Start': mass_starts,
        'Stop': mass_stops
    })

    array = extract_bin_files(data_dir=base_dir, include_fovs=[fov], panel=panel,
                                        write_tiffs=False)
    # we only care about counts, not intensities
    array = array[0, ...]
    channel_sums = np.sum(array, axis=(0, 1))

    # create df to hold output
    fovs = np.repeat(fov, len(masses))
    out_df = pd.DataFrame({'masses': masses,
                           'fov': fovs,
                           'channel_counts': channel_sums})
    out_df.to_csv(os.path.join(base_dir, fov + '_channel_counts.csv'), index=False)


def write_mph_per_mass(base_dir, fov, masses, integration_window=(-0.5, 0.5)):
    """Records the mean pulse height (MPH) per mass for the specified FOV

    Args:
        base_dir (str): the directory containing the FOV
        fov (str): the name of the fov to extract
        masses (list): the list of masses to extract MPH from
        integration_window (tuple): start and stop offset for integrating mass peak
    """
    start_offset, stop_offset = integration_window

    # create mass-specific values
    mass_starts = [mass + start_offset for mass in masses]
    mass_stops = [mass + stop_offset for mass in masses]

    # hold computed values
    mph_vals = []

    for i in range(len(masses)):
        mph_vals.append(get_histograms_per_tof(data_dir=base_dir, mass))



    