import os

import numpy as np
import tempfile

import skimage.io as io

from toffy import rosetta
import toffy.rosetta_test_cases as test_cases
from ark.utils import test_utils
from ark.utils.load_utils import load_imgs_from_tree

from ark.utils.io_utils import list_folders, list_files

import pytest
from pytest_cases import parametrize_with_cases

parametrize = pytest.mark.parametrize


def test_compensate_matrix_simple():
    inputs = np.ones((2, 40, 40, 4))

    # each channel is an increasing multiple of original
    inputs[0, :, :, 1] *= 2
    inputs[0, :, :, 2] *= 3
    inputs[0, :, :, 3] *= 4

    # second fov is 10x greater than first
    inputs[1] = inputs[0] * 10

    # define coefficient matrix; each channel has a 2x higher multiplier than previous
    coeffs = np.array([[0.01, 0, 0, 0.02], [0.02, 0, 0, 0.040],
                       [0.04, 0, 0, 0.08], [0.08, 0, 0, 0.16]])

    # calculate amount that should be removed from first channel
    total_comp = (coeffs[0, 0] * inputs[0, 0, 0, 0] + coeffs[1, 0] * inputs[0, 0, 0, 1] +
                  coeffs[2, 0] * inputs[0, 0, 0, 2] + coeffs[3, 0] * inputs[0, 0, 0, 3])

    out = rosetta.compensate_matrix_simple(inputs, coeffs)

    # non-affected channels are identical
    assert np.all(out[:, :, :, 1:-1] == inputs[:, :, :, 1:-1])

    # first channel is changed by baseline amount
    assert np.all(out[0, :, :, 0] == inputs[0, :, :, 0] - total_comp)

    # first channel in second fov is changed by baseline amount * 10 due to fov multiplier
    assert np.all(out[1, :, :, 0] == inputs[1, :, :, 0] - total_comp * 10)

    # last channel is changed by baseline amount * 2 due to multiplier in coefficient matrix
    assert np.all(out[0, :, :, -1] == inputs[0, :, :, -1] - total_comp * 2)

    # last channel in second fov is changed by baseline * 2 * 10 due to fov and coefficient
    assert np.all(out[1, :, :, -1] == inputs[1, :, :, -1] - total_comp * 10 * 2)


@parametrize('gaus_rad', [0, 1, 2])
@parametrize('save_format', ['raw', 'normalized', 'both'])
@parametrize_with_cases('panel_info', cases=test_cases.CompensateImageDataPanel)
@parametrize_with_cases('comp_mat', cases=test_cases.CompensateImageDataMat)
def test_compensate_image_data(gaus_rad, save_format, panel_info, comp_mat):
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
        comp_mat_path = os.path.join(data_dir, 'comp_mat.csv')
        comp_mat.to_csv(comp_mat_path)

        # call function
        rosetta.compensate_image_data(data_dir, output_dir, comp_mat_path, panel_info_path,
                                      save_format, gaus_rad=gaus_rad)

        # all folders created
        output_folders = list_folders(output_dir)
        assert set(fovs) == set(output_folders)

        # determine output directory structure
        format_folders = ['raw', 'normalized']
        if save_format in format_folders:
            format_folders = [save_format]

        for folder in format_folders:
            # check that all files were created
            output_files = list_files(os.path.join(output_dir, fovs[0], folder), '.tif')
            output_files = [chan.split('.tif')[0] for chan in output_files]
            assert set(output_files) == set(panel_info['targets'].values)

            output_data = load_imgs_from_tree(data_dir=output_dir, img_sub_folder=folder)
            assert np.issubdtype(output_data.dtype, np.floating)

            # all channels are smaller than original
            for i in range(output_data.shape[0]):
                for j in range(output_data.shape[-1]):
                    assert np.sum(output_data.values[i, :, :, j]) <= \
                           np.sum(data_xr.values[i, :, :, j])


@parametrize('dir_num', [2, 3])
def test_create_tiled_comparison(dir_num):
    with tempfile.TemporaryDirectory() as top_level_dir:
        num_chans = 3
        num_fovs = 4

        output_dir = os.path.join(top_level_dir, 'output_dir')
        os.makedirs(output_dir)
        dir_names = ['input_dir_{}'.format(i) for i in range(dir_num)]

        # create matching input directories
        for input_dir in dir_names:
            full_path = os.path.join(top_level_dir, input_dir)
            os.makedirs(full_path)

            fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs, num_chans=num_chans)
            filelocs, data_xr = test_utils.create_paired_xarray_fovs(
                full_path, fovs, chans, img_shape=(10, 10), fills=True)

        # pass full paths to function
        paths = [os.path.join(top_level_dir, img_dir) for img_dir in dir_names]
        rosetta.create_tiled_comparison(paths, output_dir)

        # check that each tiled image was created
        for i in range(num_chans):
            chan_name = 'chan{}_comparison.tiff'.format(i)
            chan_img = io.imread(os.path.join(output_dir, chan_name))
            row_len = num_fovs * 10
            col_len = dir_num * 10
            assert chan_img.shape == (col_len, row_len)

        # check that directories with different images raises error
        for i in range(num_fovs):
            os.remove(os.path.join(top_level_dir, dir_names[0], 'fov{}'.format(i), 'chan0.tiff'))
        with pytest.raises(ValueError):
            rosetta.create_tiled_comparison(paths, output_dir)


def test_add_source_channel_to_tiled_image():
    with tempfile.TemporaryDirectory() as top_level_dir:
        num_fovs = 5
        num_chans = 4
        im_size = 10

        # create directory containing raw images
        raw_dir = os.path.join(top_level_dir, 'raw_dir')
        os.makedirs(raw_dir)

        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs, num_chans=num_chans)
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            raw_dir, fovs, chans, img_shape=(im_size, im_size), fills=True)

        # create directory containing stitched images
        tiled_shape = (im_size * 3, im_size * num_fovs)
        tiled_dir = os.path.join(top_level_dir, 'tiled_dir')
        os.makedirs(tiled_dir)
        for i in range(2):
            vals = np.random.rand(im_size * 3 * im_size * num_fovs).reshape(tiled_shape)
            io.imsave(os.path.join(tiled_dir, 'tiled_image_{}.tiff'.format(i)), vals)

        output_dir = os.path.join(top_level_dir, 'output_dir')
        os.makedirs(output_dir)
        rosetta.add_source_channel_to_tiled_image(raw_img_dir=raw_dir, tiled_img_dir=tiled_dir,
                                                  output_dir=output_dir, source_channel='chan1')
        # get vals of row that was added to top
        vals = data_xr.loc[:, :, :, 'chan1'].values
        vals_list = [vals[i, ...] for i in range(vals.shape[0])]
        vals_row = np.concatenate(vals_list, axis=1)

        # top portion of each image should be the same
        tiled_images = list_files(output_dir)
        for im_name in tiled_images:
            image = io.imread(os.path.join(output_dir, im_name))
            assert np.array_equal(image[:im_size, :], vals_row)
