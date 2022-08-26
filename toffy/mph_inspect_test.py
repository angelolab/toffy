import os

import pytest
import tempfile

import numpy as np
from pathlib import Path

from toffy import mph_inspect


def test_bin_array():
    sample_intensities = np.array([range(1, 101)])
    bin_factor = 20
    expected_binned = np.array([230, 630, 1030, 1430])
    binned_intensities = mph_inspect.bin_array(sample_intensities, bin_factor)

    # test successful binning
    assert (binned_intensities == expected_binned).all()


def create_hist_per_tof_data(bin_file_dir, fov, target, panel):
    sample_intensities = np.array([range(1, 101)])
    sample_pulse_counts = np.array([range(1, 101)])

    return 0, sample_intensities, sample_pulse_counts


def test_compute_intensities(mocker):
    mocker.patch('toffy.mph_inspect.get_histograms_per_tof', create_hist_per_tof_data)

    mass = 98
    mass_start = 97.5
    mass_stop = 98.5

    with tempfile.TemporaryDirectory() as tmp_dir:
        open(os.path.join(tmp_dir, "fov-1.bin"), 'w')
        mph_data = mph_inspect.compute_intensities(tmp_dir, ["fov-1"], mass, mass_start,
                                                   mass_stop, bin_factor=20)

        # test remaining columns
        assert mph_data['fov'][0] == "fov-1"
        assert mph_data['median_intensity'][0] == 70


def test_visualize_intensity_data(mocker):
    bad_path = os.path.join('data', 'not_bin_file_dir')
    bin_file_dir = good_path = os.path.join(Path(__file__).parent, "data", "combined")
    mass = 98
    mass_start = 97.5
    mass_stop = 98.5

    # bad path should raise an error
    with pytest.raises(ValueError):
        mph_inspect.visualize_intensity_data(bad_path, mass, mass_start, mass_stop)

    # bad fov name should raise error
    bad_fov_list = ['fov-1-scan-1', 'bad_fov']
    with pytest.raises(ValueError, match="Not all values given in list"):
        mph_inspect.visualize_intensity_data(bin_file_dir, mass, mass_start, mass_stop,
                                             bad_fov_list)

    # successful function should raise no errors
    mocker.patch('toffy.mph_inspect.get_histograms_per_tof', create_hist_per_tof_data)
    with tempfile.TemporaryDirectory() as tmp_dir:
        open(os.path.join(tmp_dir, "fov-1.bin"), 'w')
        mph_inspect.visualize_intensity_data(tmp_dir, mass, mass_start, mass_stop, bin_factor=20)
