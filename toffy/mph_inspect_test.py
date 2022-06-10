import os
import pytest_mock
import tempfile

import numpy as np
import pandas as pd

from toffy import mph_inspect


def test_bin_array():
    sample_intensities = np.array([range(1, 101)])
    bin_factor = 20
    expected_binned = np.array([230, 630, 1030, 1430])
    binned_intensities = mph_inspect.bin_array(sample_intensities, bin_factor)

    assert (binned_intensities == expected_binned).all()


def sample_data(bin_file_dir, fov, target, panel):
    sample_intensities = np.array([range(1, 101)])
    sample_pulse_counts = np.array([range(1, 101)])
    print("worked")
    return 0, sample_intensities, sample_pulse_counts


def test_compute_mph_intensities(mocker):
    mocker.patch('toffy.mph_inspect.get_histograms_per_tof', sample_data)

    mass = 98
    mass_start = 97.5
    mass_stop = 98.5
    panel = pd.DataFrame([{
        'Mass': mass,
        'Target': None,
        'Start': mass_start,
        'Stop': mass_stop,
    }])

    with tempfile.TemporaryDirectory() as tmp_dir:
        open(os.path.join(tmp_dir, "fov-1.bin"), 'w')
        #open(os.path.join(tmp_dir, "fov-2.bin"), 'w')
        mph_data = mph_inspect.compute_mph_intensities(tmp_dir, mass, mass_start, mass_stop,
                                                       bin_factor=20)
        #base_dir=os.path.join('data', 'tissue')
        #mph_data.to_csv(os.path.join(base_dir, "test.csv"))


#def test_visualize_mph_hist():
#    pass
