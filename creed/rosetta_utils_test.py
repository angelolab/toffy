import os

import numpy as np
import pandas as pd
import tempfile

from creed import rosetta_utils
from ark.utils import test_utils
from ark.utils.load_utils import load_imgs_from_tree

from ark.utils.io_utils import list_folders, list_files


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


def test_compensate_image_data():
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
        masses = ['25', '50', '101']
        d = {'masses': masses, 'targets': chans}
        panel_info = pd.DataFrame(d)
        panel_info_path = os.path.join(data_dir, 'panel_info.csv')
        panel_info.to_csv(panel_info_path, index=False)

        # create compensation matrix
        comp_mat_vals = np.random.rand(3, 3) / 100

        # channel 0  will be unchanged
        comp_mat_vals[:, 0] = 0

        # channel 2 has 3x the difference of channel 1
        comp_mat_vals[:, 2] = comp_mat_vals[:, 1] * 3

        # create compensation matrix
        comp_mat = pd.DataFrame(comp_mat_vals, columns=masses, index=masses)
        comp_mat_path = os.path.join(data_dir, 'comp_mat.csv')
        comp_mat.to_csv(comp_mat_path)

        # run with default settings
        rosetta_utils.compensate_image_data(data_dir, output_dir, comp_mat_path, panel_info_path,
                                            False, gaus_rad=0)

        # all folders created
        output_folders = list_folders(output_dir)
        assert set(fovs) == set(output_folders)

        # all channels processed
        output_files = list_files(os.path.join(output_dir, fovs[0]), '.tif')
        output_files = [chan.split('.tif')[0] for chan in output_files]
        assert set(output_files) == set(chans)
        output_data = load_imgs_from_tree(data_dir=output_dir)

        # first channel is unmodified
        assert np.all(output_data.values[:, :, :, 0] == data_xr.values[:, :, :, 0])

        # other channels are smaller than original
        for i in range(output_data.shape[0]):
            for j in range(1, output_data.shape[-1]):
                assert np.all(output_data.values[i, :, :, j] <= data_xr.values[i, :, :, j])

        # change in channel 2 is 3x change in channel 1
        for i in range(output_data.shape[0]):
            chan1_dif = np.sum(output_data.values[i, :, :, 1] - data_xr.values[i, :, :, 1])
            chan2_dif = np.sum(output_data.values[i, :, :, 2] - data_xr.values[i, :, :, 2])

            np.testing.assert_almost_equal(chan2_dif, chan1_dif * 3)
