import os

import numpy as np
import pandas as pd
import tempfile

from creed import rosetta_utils
import creed.rosetta_utils_test_cases as test_cases
from ark.utils import test_utils
from ark.utils.load_utils import load_imgs_from_tree

from ark.utils.io_utils import list_folders, list_files

import pytest
from pytest_cases import parametrize_with_cases



def test_compensate_matrix_simple():
    inputs = np.ones((2, 40, 40, 4))

    # each channel is an increasing multiple of original
    inputs[0, :, :, 1] *= 2
    inputs[0, :, :, 2] *= 3
    inputs[0, :, :, 3] *= 4

    # second fov is 10x greater than first
    inputs[1] = inputs[0] * 10

    # define coefficient matrix; each channel has a 10x higher multiplier than previous
    coeffs = np.array([[1, 0, 0, 2], [10, 0, 0, 20], [100, 0, 0, 200], [1000, 0, 0, 2000]])

    # calculate amount that should be removed from first channel
    total_comp = (coeffs[0, 0] * inputs[0, 0, 0, 0] + coeffs[1, 0] * inputs[0, 0, 0, 1] +
                  coeffs[2, 0] * inputs[0, 0, 0, 2] + coeffs[3, 0] * inputs[0, 0, 0, 3])

    out = rosetta_utils.compensate_matrix_simple(inputs, coeffs)

    # non-affected channels are identical
    assert np.all(out[:, :, :, 1:-1] == inputs[:, :, :, 1:-1])

    # first channel is changed by expected amount
    assert np.all(out[0, :, :, 0] == inputs[0, :, :, 0] - total_comp)

    # first channel in second fov is changed by expected amount
    assert np.all(out[1, :, :, 0] == inputs[1, :, :, 0] - total_comp * 10)

    # last channel is changed by expected amount
    assert np.all(out[0, :, :, -1] == inputs[0, :, :, -1] - total_comp * 2)

    # last channel in second fov is changed by expected amount
    assert np.all(out[1, :, :, -1] == inputs[1, :, :, -1] - total_comp * 10 * 2)


@pytest.mark.parametrize('gaus_rad', [0, 1, 2])
@parametrize_with_cases('panel_info', cases=test_cases.CompensateImageDataPanel)
def test_compensate_image_data(gaus_rad, panel_info):
    with tempfile.TemporaryDirectory() as top_level_dir:
        data_dir = os.path.join(top_level_dir, 'data_dir')
        output_dir = os.path.join(top_level_dir, 'output_dir')

        os.makedirs(data_dir)
        os.makedirs(output_dir)

        # make fake data for testing
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=3)
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            data_dir, fovs, chans, img_shape=(10, 10), fills=True)

        # create panel info csv
        panel_info_path = os.path.join(data_dir, 'panel_info.csv')
        panel_info.to_csv(panel_info_path, index=False)

        # create compensation matrix
        comp_mat_vals = np.random.rand(3, 3) / 100
        comp_mat = pd.DataFrame(comp_mat_vals, columns=['25', '50', '101'], index=['25', '50', '101'])
        comp_mat_path = os.path.join(data_dir, 'comp_mat.csv')
        comp_mat.to_csv(comp_mat_path)

        # run with default settings
        rosetta_utils.compensate_image_data(data_dir, output_dir, comp_mat_path, panel_info_path,
                                            'raw', gaus_rad=gaus_rad)

        # all folders created
        output_folders = list_folders(output_dir)
        assert set(fovs) == set(output_folders)

        # all channels processed
        output_files = list_files(os.path.join(output_dir, fovs[0], 'raw'), '.tif')
        output_files = [chan.split('.tif')[0] for chan in output_files]
        assert set(output_files) == set(panel_info['targets'].values)
        output_data = load_imgs_from_tree(data_dir=output_dir, img_sub_folder='raw')

        # all channels are smaller than original
        for i in range(output_data.shape[0]):
            for j in range(output_data.shape[-1]):
                assert np.sum(output_data.values[i, :, :, j]) <= np.sum(data_xr.values[i, :, :, j])

