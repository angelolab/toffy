import os

import numpy as np
import pandas as pd
import pytest
import tempfile
import xarray as xr

from toffy import watcher_functions


def mocked_extract_bin_file(data_dir, include_fovs, panel, out_dir, intensities):
    mass_num = len(panel)

    base_img = np.ones((3, 4, 4))

    all_imgs = []
    for i in range(1, mass_num + 1):
        all_imgs.append(base_img * i)

    out_img = np.stack(all_imgs, axis=-1)

    out_img = np.expand_dims(out_img, axis=0)

    out_array = xr.DataArray(data=out_img,
                             coords=[
                                [include_fovs[0]],
                                ['pulse', 'intensity', 'area'],
                                np.arange(base_img.shape[1]),
                                np.arange(base_img.shape[2]),
                                panel['Target'].values,
                             ],
                             dims=['fov', 'type', 'x', 'y', 'channel'])
    return out_array


def mocked_pulse_height(data_dir, fov, panel, channel):
    mass = panel['Mass'].values[0]
    return mass * 2


def test_write_counts_per_mass(mocker):
    with tempfile.TemporaryDirectory() as temp_dir:
        masses = [88, 89, 90]
        expected_counts = [16 * i for i in range(1, len(masses) + 1)]
        mocker.patch('toffy.watcher_functions.extract_bin_files', mocked_extract_bin_file)

        watcher_functions.write_counts_per_mass(base_dir=temp_dir, fov='fov1',
                                                masses=masses)
        output = pd.read_csv(os.path.join(temp_dir, 'fov1_channel_counts.csv'))
        assert len(output) == len(masses)
        assert set(output['mass'].values) == set(masses)
        assert set(output['channel_count'].values) == set(expected_counts)


def test_write_mph_per_mass(mocker):
    with tempfile.TemporaryDirectory() as temp_dir:
        masses = [88, 89, 90]
        mocker.patch('toffy.watcher_functions.get_median_pulse_height', mocked_pulse_height)

        watcher_functions.write_mph_per_mass(base_dir=temp_dir, fov='fov1', masses=masses)
        output = pd.read_csv(os.path.join(temp_dir, 'fov1_pulse_heights.csv'))
        assert len(output) == len(masses)
        assert set(output['mass'].values) == set(masses)
        assert np.all(output['pulse_height'].values == output['mass'].values * 2)
