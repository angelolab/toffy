import os
import pandas as pd
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

        # create compensation matrix
        comp_mat_path = os.path.join(data_dir, 'comp_mat.csv')
        comp_mat.to_csv(comp_mat_path)

        # call function
        rosetta.compensate_image_data(data_dir, output_dir, comp_mat_path, panel_info,
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
            assert set(output_files) == set(panel_info['Target'].values)

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
                full_path, fovs, chans, img_shape=(10, 10), fills=True, sub_dir='normalized')

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
            os.remove(os.path.join(top_level_dir, dir_names[0], 'fov{}'.format(i),
                                   'normalized/chan0.tiff'))
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

        # each image should now have an extra row added on top
        tiled_images = list_files(output_dir)
        for im_name in tiled_images:
            image = io.imread(os.path.join(output_dir, im_name))
            assert image.shape == (tiled_shape[0] + im_size, tiled_shape[1])


@parametrize('folders', [None, ['run_0']])
@parametrize('overwrite', [True, False])
def test_replace_with_intensity_image(overwrite, folders):
    with tempfile.TemporaryDirectory() as top_level_dir:
        num_fovs = 2
        num_chans = 2
        im_size = 10
        num_dirs = 3

        for dir in range(num_dirs):
            # create directory containing raw images
            run_dir = os.path.join(top_level_dir, 'run_{}'.format(dir))
            os.makedirs(run_dir)

            fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs, num_chans=num_chans)
            chans = [chan + '_intensity' for chan in chans]
            filelocs, data_xr = test_utils.create_paired_xarray_fovs(
                run_dir, fovs, chans, img_shape=(im_size, im_size), fills=True,
                sub_dir='intensities')

        rosetta.replace_with_intensity_image(base_dir=top_level_dir, channel='chan1',
                                             replace=overwrite, folders=folders)

        # loop through all fovs in all directories to check that correct image was written
        for dir in range(num_dirs):
            for fov in range(num_fovs):
                if folders is not None and dir > 0:
                    # no image should be copied over for these
                    files = list_files(os.path.join(top_level_dir, 'run_{}'.format(dir),
                                                    'fov{}'.format(fov)))
                    assert len(files) == 0
                else:
                    if overwrite:
                        suffix = '.tiff'
                    else:
                        suffix = '_intensity.tiff'
                    file = os.path.join(top_level_dir, 'run_{}'.format(dir),
                                        'fov{}'.format(fov), 'chan1' + suffix)
                    assert os.path.exists(file)


def test_create_rosetta_matrices():
    # step 1: create rosetta matrix
    # step 2: save as csv
    with tempfile.TemporaryDirectory() as temp_dir:
        # Pulling in channel names
        test_channels = [23, 48, 56, 69, 71, 89, 113, 115, 117, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
                         152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
                         171, 172, 173, 174, 175, 176, 181, 197]

        # Creating random matrix to test
        rand_np_matrix = np.random.randint(1, 50, size=[47, 47])
        random_matrix = pd.DataFrame(rand_np_matrix, index=test_channels, columns=test_channels)
        output_path = os.path.join(temp_dir, 'Random_matrix.csv')
        random_matrix.to_csv(output_path)

        # Checking channels = None
        multipliers = [3]
        create_rosetta_matrices(output_path, temp_dir, multipliers)

        rosetta_path = os.path.join(temp_dir, 'Rosetta_Titration%s.csv' % (str(multipliers[0])))
        test_matrix = pd.read_csv(rosetta_path, index_col=0).astype(int)  # grabs output of create_rosetta_matrices
        validation = (test_matrix / multipliers[0])

        # confirm all channels scaled by multiplier
        assert np.array_equal(random_matrix, validation)

        # Checking Specific Channels
        multipliers = [4]
        channels = [197]
        create_rosetta_matrices(output_path, temp_dir, multipliers, channels)

        rosetta_path_2 = os.path.join(temp_dir, 'Rosetta_Titration%s.csv' % (str(multipliers[0])))
        test_matrix_2 = pd.read_csv(rosetta_path_2, index_col=0).astype(int)  # grabs output of create_rosetta_matrices

        test_index = test_channels.index(channels[0])  # row index of input channel
        validation = (test_matrix_2.iloc[test_index, :] / multipliers[0]).astype(int)  # divides row by multiplier
        random_matrix_channel = random_matrix.iloc[test_index]  # row of random matrix at channel

        # confirm input channel is scaled by multiplier
        assert np.array_equal(random_matrix_channel, validation)

        # confirm non-specified channels are unchanged
        for i in test_channels:
            if i not in channels:
                channel_index = test_channels.index(i)
                assert np.array_equal(test_matrix_2.iloc[channel_index, :], random_matrix.iloc[channel_index, :])
