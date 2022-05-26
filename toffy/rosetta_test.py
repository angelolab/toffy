import copy
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
from toffy.rosetta import create_rosetta_matrices
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

    out_indices = np.arange(inputs.shape[-1])
    out = rosetta._compensate_matrix_simple(inputs, coeffs, out_indices)

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

    # don't generate output for first channel
    out_indices = out_indices[1:]
    out = rosetta._compensate_matrix_simple(inputs, coeffs, out_indices)

    # non-affected channels are identical
    assert np.all(out[:, :, :, :-1] == inputs[:, :, :, 1:-1])

    # last channel is changed by baseline amount * 2 due to multiplier in coefficient matrix
    assert np.all(out[0, :, :, -1] == inputs[0, :, :, -1] - total_comp * 2)

    # last channel in second fov is changed by baseline * 2 * 10 due to fov and coefficient
    assert np.all(out[1, :, :, -1] == inputs[1, :, :, -1] - total_comp * 10 * 2)


def test_validate_inputs():
    with tempfile.TemporaryDirectory() as top_level_dir:
        data_dir = os.path.join(top_level_dir, 'data_dir')
        os.makedirs(data_dir)

        # make fake data for testing
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=3)
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            data_dir, fovs, chans, img_shape=(10, 10), fills=True)

        # generate default, correct values for all parameters
        masses = [71, 76, 101]
        comp_mat_vals = np.random.rand(len(masses), len(masses)) / 100
        comp_mat = pd.DataFrame(comp_mat_vals, columns=masses, index=masses)

        acquired_masses = copy.copy(masses)
        acquired_targets = copy.copy(chans)
        input_masses = copy.copy(masses)
        output_masses = copy.copy(masses)
        all_masses = copy.copy(masses)
        save_format = 'raw'
        raw_data_sub_folder = ''
        batch_size = 1
        gaus_rad = 1

        input_dict = {'raw_data_dir': data_dir, 'comp_mat': comp_mat,
                      'acquired_masses': acquired_masses, 'acquired_targets': acquired_targets,
                      'input_masses': input_masses,
                      'output_masses': output_masses, 'all_masses': all_masses, 'fovs': fovs,
                      'save_format': save_format, 'raw_data_sub_folder': raw_data_sub_folder,
                      'batch_size': batch_size, 'gaus_rad': gaus_rad}

        # check that masses are sorted
        input_dict_disorder = copy.copy(input_dict)
        input_dict_disorder['acquired_masses'] = masses[1:] + masses[:1]
        with pytest.raises(ValueError, match='Masses must be sorted'):
            rosetta.validate_inputs(**input_dict_disorder)

        # check that all masses are present
        input_dict_missing = copy.copy(input_dict)
        input_dict_missing['acquired_masses'] = masses[1:]
        with pytest.raises(ValueError, match='acquired masses and list compensation masses'):
            rosetta.validate_inputs(**input_dict_missing)

        # check that images and channels are the same
        input_dict_img_name = copy.copy(input_dict)
        input_dict_img_name['acquired_targets'] = chans + ['chan15']
        with pytest.raises(ValueError, match='given in list listed channels'):
            rosetta.validate_inputs(**input_dict_img_name)

        # check that input masses are valid
        input_dict_input_mass = copy.copy(input_dict)
        input_dict_input_mass['input_masses'] = masses + [17]
        with pytest.raises(ValueError, match='list input masses'):
            rosetta.validate_inputs(**input_dict_input_mass)

        # check that output masses are valid
        input_dict_output_mass = copy.copy(input_dict)
        input_dict_output_mass['output_masses'] = masses + [17]
        with pytest.raises(ValueError, match='list output masses'):
            rosetta.validate_inputs(**input_dict_output_mass)

        # check that comp_mat has no NAs
        input_dict_na = copy.copy(input_dict)
        comp_mat_na = copy.copy(comp_mat)
        comp_mat_na.iloc[0, 2] = np.nan
        input_dict_na['comp_mat'] = comp_mat_na
        with pytest.raises(ValueError, match='no missing values'):
            rosetta.validate_inputs(**input_dict_na)

        # check that save_format is valid
        input_dict_save_format = copy.copy(input_dict)
        input_dict_save_format['save_format'] = 'bad'
        with pytest.raises(ValueError, match='list save format'):
            rosetta.validate_inputs(**input_dict_save_format)

        # check that batch_size is valid
        input_dict_batch_size = copy.copy(input_dict)
        input_dict_batch_size['batch_size'] = 1.5
        with pytest.raises(ValueError, match='batch_size parameter'):
            rosetta.validate_inputs(**input_dict_batch_size)

        # check that gaus_rad is valid
        input_dict_gaus_rad = copy.copy(input_dict)
        input_dict_gaus_rad['gaus_rad'] = -1
        with pytest.raises(ValueError, match='gaus_rad parameter'):
            rosetta.validate_inputs(**input_dict_gaus_rad)


def test_flat_field_correction():
    input_img = np.random.rand(10, 10)
    corrected_img = rosetta.flat_field_correction(img=input_img)

    assert corrected_img.shape == input_img.shape
    assert not np.array_equal(corrected_img, input_img)


def test_get_masses_from_channel_names():
    targets = ['chan1', 'chan2', 'chan3']
    masses = [1, 2, 3]
    test_df = pd.DataFrame({'Target': targets,
                            'Mass': masses})

    all_masses = rosetta.get_masses_from_channel_names(targets, test_df)
    assert np.array_equal(masses, all_masses)

    subset_masses = rosetta.get_masses_from_channel_names(targets[:2], test_df)
    assert np.array_equal(masses[:2], subset_masses)

    with pytest.raises(ValueError, match='channel names'):
        rosetta.get_masses_from_channel_names(['chan4'], test_df)


@parametrize('output_masses', [None, [25, 50, 101], [25, 50]])
@parametrize('input_masses', [None, [25, 50, 101], [25, 50]])
@parametrize('gaus_rad', [0, 1, 2])
@parametrize('save_format', ['raw', 'normalized', 'both'])
@parametrize_with_cases('panel_info', cases=test_cases.CompensateImageDataPanel)
@parametrize_with_cases('comp_mat', cases=test_cases.CompensateImageDataMat)
def test_compensate_image_data(output_masses, input_masses, gaus_rad, save_format, panel_info,
                               comp_mat):
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
                                      input_masses=input_masses, output_masses=output_masses,
                                      save_format=save_format, gaus_rad=gaus_rad,
                                      ffc_channels=['chan1'])

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

            if output_masses is None or len(output_masses) == 3:
                assert set(output_files) == set(panel_info['Target'].values)
            else:
                assert set(output_files) == set(panel_info['Target'].values[:-1])

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

        # check that directories with different images are okay if overlapping channels specified
        for i in range(num_fovs):
            os.remove(os.path.join(top_level_dir, dir_names[1], 'fov{}'.format(i),
                                   'normalized/chan0.tiff'))

        # no error raised if subset directory is specified
        rosetta.create_tiled_comparison(paths, output_dir, channels=['chan1', 'chan2'])

        # but one is raised if no subset directory is specified
        with pytest.raises(ValueError, match='1 of 1'):
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


@parametrize('fovs', [None, ['fov1']])
@parametrize('replace', [True, False])
def test_replace_with_intensity_image(replace, fovs):
    with tempfile.TemporaryDirectory() as top_level_dir:

        # create directory containing raw images
        run_dir = os.path.join(top_level_dir, 'run_dir')
        os.makedirs(run_dir)

        fov_names, chans = test_utils.gen_fov_chan_names(num_fovs=2, num_chans=2)
        chans = [chan + '_intensity' for chan in chans]
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            run_dir, fov_names, chans, img_shape=(10, 10), fills=True,
            sub_dir='intensities')

        rosetta.replace_with_intensity_image(run_dir=run_dir, channel='chan1',
                                             replace=replace, fovs=fovs)

        # loop through all fovs to check that correct image was written
        for current_fov in range(2):
            if fovs is not None and current_fov == 0:
                # this fov was skipped, no images should be present here
                files = list_files(os.path.join(run_dir, 'fov0'))
                assert len(files) == 0
            else:
                # ensure correct extension is present
                if replace:
                    suffix = '.tiff'
                else:
                    suffix = '_intensity.tiff'
                file = os.path.join(run_dir, 'fov{}'.format(current_fov), 'chan1' + suffix)
                assert os.path.exists(file)


def test_remove_sub_dirs():
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs = ['fov1', 'fov2', 'fov3']
        sub_dirs = ['sub1', 'sub2', 'sub3']

        # make directory structure
        for fov in fovs:
            os.makedirs(os.path.join(temp_dir, fov))
            for sub_dir in sub_dirs:
                os.makedirs(os.path.join(temp_dir, fov, sub_dir))

        rosetta.remove_sub_dirs(run_dir=temp_dir, sub_dirs=sub_dirs[1:], fovs=fovs[:-1])

        # check that last fov has all sub_dirs, all other fovs have appropriate sub_dirs removed
        for fov in fovs:
            if fov == fovs[-1]:
                expected_dirs = sub_dirs
            else:
                expected_dirs = sub_dirs[:1]

            for sub_dir in sub_dirs:
                if sub_dir in expected_dirs:
                    assert os.path.exists(os.path.join(temp_dir, fov, sub_dir))
                else:
                    assert not os.path.exists(os.path.join(temp_dir, fov, sub_dir))


def test_create_rosetta_matrices():
    with tempfile.TemporaryDirectory() as temp_dir:

        # create baseline rosetta matrix
        test_channels = [23, 71, 89, 113, 141, 142, 143]
        base_matrix = np.random.randint(1, 50, size=[len(test_channels), len(test_channels)])
        base_rosetta = pd.DataFrame(base_matrix, index=test_channels, columns=test_channels)
        base_rosetta_path = os.path.join(temp_dir, 'rosetta_matrix.csv')
        base_rosetta.to_csv(base_rosetta_path)

        # validate output when all channels are included
        multipliers = [0.5, 2, 4]
        create_rosetta_matrices(base_rosetta_path, temp_dir, multipliers)

        for multiplier in multipliers:
            rosetta_path = os.path.join(temp_dir, 'rosetta_matrix_mult_%s.csv'
                                        % (str(multiplier)))
            # grabs output of create_rosetta_matrices
            test_matrix = pd.read_csv(rosetta_path, index_col=0)
            rescaled = (test_matrix / multiplier)

            # confirm all channels scaled by multiplier
            assert np.array_equal(base_rosetta, rescaled)

        # validate output for specific channels
        mod_channels = [113, 142]
        create_rosetta_matrices(base_rosetta_path, temp_dir, multipliers, mod_channels)

        # create mask specifying which channels will change
        change_idx = np.isin(test_channels, mod_channels)

        for mult in multipliers:
            mult_vec = np.ones(len(test_channels))
            mult_vec[change_idx] = mult

            # grabs output of create_rosetta_matrices
            rosetta_path = os.path.join(temp_dir, 'rosetta_matrix_mult_%s.csv' % (str(mult)))
            test_matrix = pd.read_csv(rosetta_path, index_col=0)

            rescaled = test_matrix.divide(mult_vec, axis='index')
            assert np.array_equal(base_rosetta, rescaled)

        # check that error is raised when non-numeric rosetta_matrix is passed
        bad_matrix = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': ['a', 'b', 'c']})
        bad_matrix_path = os.path.join(temp_dir, 'bad_rosetta_matrix.csv')
        bad_matrix.to_csv(bad_matrix_path)

        with pytest.raises(ValueError, match='include only numeric'):
            create_rosetta_matrices(bad_matrix_path, temp_dir, multipliers)
