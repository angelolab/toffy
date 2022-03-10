import copy
import os
import json
import shutil

import numpy as np
import pandas as pd

import skimage.io as io
from scipy.ndimage import gaussian_filter

from ark.utils.load_utils import load_imgs_from_tree, load_imgs_from_dir
from ark.utils.io_utils import list_folders, validate_paths, list_files
from ark.utils.misc_utils import verify_same_elements, verify_in_list


def transform_compensation_json(json_path, comp_mat_path):
    """Converts the JSON file from ionpath into a compensation matrix

    Args:
        json_path: path to json file
        comp_mat_path: path to comp matrix

    returns:
        pd.DataTable: matrix with sources channels as rows and target channels as columns"""

    with open(json_path) as f:
        data = json.load(f)['Data']

    comp_mat = pd.read_csv(comp_mat_path, index_col=0)

    for i in range(len(data)):
        current_data = data[i]
        source = current_data['DonorMass']
        target = current_data['RecipientMass']
        val = current_data['Percent'] / 100
        comp_mat.loc[source, str(target)] = val

    comp_mat.to_csv(comp_mat_path)


def _compensate_matrix_simple(raw_inputs, comp_coeffs, out_indices):
    """Perform compensation on the raw data using the supplied compensation values

    Args:
        raw_inputs (numpy.ndarray):
            array with shape [fovs, rows, cols, channels] containing the image data
        comp_coeffs (numpy.ndarray):
            2D array of coefficients with source channels as rows and targets as columns
        out_indices (numpy.ndarray):
            which indices to generate compensated outputs for

    returns:
        numpy.ndarray: compensated copy of the raw inputs"""

    outputs = np.copy(raw_inputs)

    # loop over each channel and construct compensation values
    for chan in out_indices:
        chan_coeffs = comp_coeffs[:, chan]

        # convert from 1D to 4D for broadcasting
        chan_coeffs = np.reshape(chan_coeffs, (1, 1, 1, len(chan_coeffs)))

        # broadcast across entire dataset and collapse into single set of values
        chan_vals = raw_inputs * chan_coeffs
        chan_vals = np.sum(chan_vals, axis=-1)

        # subtract compensated values from target channel
        outputs[..., chan] -= chan_vals

    # set negative values to zero
    outputs = np.where(outputs > 0, outputs, 0)

    # subset on specified indices
    outputs = outputs[..., out_indices]

    return outputs


def validate_inputs(raw_data_dir, comp_mat, acquired_masses, acquired_targets, input_masses,
                    output_masses, all_masses, fovs, save_format, batch_size, gaus_rad):
    """Helper function to validate inputs for compensate_image_data

    Args:
        raw_data_dir (str): path to raw data
        comp_mat (pd.DataFrame): compensation matrix
        acquired_masses (list): masses in the supplied panel
        acquired_targets (list): targets in the supplied panel
        input_masses (list): masses to use for compensation
        output_masses (list): masses to compensate
        all_masses (list): masses in the compensation matrix
        fovs (list): fovs in the raw_data_dir
        save_format (str): format to save the data
        batch_size (int): number of images to process concurrently
        gaus_rad (int): radius for smoothing"""

    # make sure panel is in increasing order
    if not np.all(acquired_masses == sorted(acquired_masses)):
        raise ValueError("Masses must be sorted numerically in the panel_info file")

    # make sure channels in comp matrix are same as those in panel csv
    verify_same_elements(acquired_masses=acquired_masses, compensation_masses=all_masses)

    # check first FOV to make sure all channels are present
    test_data = load_imgs_from_tree(data_dir=raw_data_dir, fovs=fovs[0:1],
                                    channels=acquired_targets, dtype='float32')

    verify_same_elements(image_files=test_data.channels.values, listed_channels=acquired_targets)

    # make sure supplied masses are present
    if input_masses is not None:
        verify_in_list(input_masses=input_masses, compensation_masses=all_masses)

    if output_masses is not None:
        verify_in_list(output_masses=output_masses, compensation_masses=all_masses)

    # make sure compensation matrix has valid values
    if comp_mat.isna().values.any():
        raise ValueError('Compensation matrix must contain a value for every field; check to '
                         'make sure there are no missing values')

    # check for valid save_formats
    allowed_formats = ['raw', 'normalized', 'both']
    verify_in_list(save_format=save_format, allowed_formats=allowed_formats)

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size parameter must be a positive integer')

    if not isinstance(gaus_rad, int) or gaus_rad < 0:
        raise ValueError('gaus_rad parameter must be a non-negative integer')


def compensate_image_data(raw_data_dir, comp_data_dir, comp_mat_path, panel_info,
                          input_masses=None, output_masses=None, save_format='normalized',
                          batch_size=10, gaus_rad=1, norm_const=100):
    """Function to compensate MIBI data with a flow-cytometry style compensation matrix

    Args:
        raw_data_dir: path to directory containing raw images
        comp_data_dir: path to directory where compensated images will be saved
        comp_mat_path: path to compensation matrix, nxn with channel labels
        panel_info: array with information about the panel
        input_masses (list): masses from the compensation matrix to use for compensation. If None,
            all masses will be used
        output_masses (list): masses from the compensation matrix that will have tifs generated. If
            None, all masses will be used
        save_format: flag to control how the processed tifs are saved. Must be one of:
            'raw': Direct output from the compensation matrix corresponding to number of ion events
                detected per pixel. These will not be viewable in many image processing programs
            'normalized': all images are divided by 100 to enable visualization. This transform
                has no effect on downstream analysis as it preserves relative expression values
            'both': saves both 'raw' and 'normalized' images
        batch_size: number of images to process at a time
        gaus_rad: radius for blurring image data. Passing 0 will result in no blurring
        norm_const: constant used for normalization if save_format == 'normalized'
    """

    validate_paths([raw_data_dir, comp_data_dir, comp_mat_path],
                   data_prefix=False)

    # get list of all fovs
    fovs = list_folders(raw_data_dir, substrs=['fov'])

    # load csv files
    comp_mat = pd.read_csv(comp_mat_path, index_col=0)
    acquired_masses = panel_info['Mass'].values
    acquired_targets = panel_info['Target'].values
    all_masses = comp_mat.columns.values.astype('int')

    validate_inputs(raw_data_dir, comp_mat, acquired_masses, acquired_targets, input_masses,
                    output_masses, all_masses, fovs, save_format, batch_size, gaus_rad)

    # set unused masses to zero
    if input_masses is not None:
        zero_idx = ~np.isin(all_masses, input_masses)
        comp_mat.loc[zero_idx, :] = 0

    if output_masses is None:
        out_indices = np.arange(len(all_masses))
    else:
        out_indices = np.where(np.isin(all_masses, output_masses))[0]

    # loop over each set of FOVs in the batch
    for i in range(0, len(fovs), batch_size):
        print("Processing image {}".format(i))

        # load batch of fovs
        batch_fovs = fovs[i: i + batch_size]
        batch_data = load_imgs_from_tree(data_dir=raw_data_dir, fovs=batch_fovs,
                                         channels=acquired_targets, dtype='float32')

        # blur data
        if gaus_rad > 0:
            for j in range(batch_data.shape[0]):
                for k in range(batch_data.shape[-1]):
                    batch_data[j, :, :, k] = gaussian_filter(batch_data[j, :, :, k],
                                                             sigma=gaus_rad)

        comp_data = _compensate_matrix_simple(raw_inputs=batch_data,
                                              comp_coeffs=comp_mat.values,
                                              out_indices=out_indices)

        # save data
        for j in range(batch_data.shape[0]):
            fov_folder = os.path.join(comp_data_dir, batch_data.fovs.values[j])
            os.makedirs(fov_folder)

            # create directories for saving tifs
            if save_format in ['normalized', 'both']:
                norm_folder = os.path.join(fov_folder, 'normalized')
                os.makedirs(norm_folder)

            if save_format in ['raw', 'both']:
                raw_folder = os.path.join(fov_folder, 'raw')
                os.makedirs(raw_folder)

            # this may be only a subset of masses, based on output_masses
            for idx, val in enumerate(out_indices):
                channel_name = batch_data.channels.values[val] + '.tiff'

                # save tifs to appropriate directories
                if save_format in ['normalized', 'both']:
                    save_path = os.path.join(norm_folder, channel_name)
                    io.imsave(save_path, comp_data[j, :, :, idx] / norm_const,
                              check_contrast=False)

                if save_format in ['raw', 'both']:
                    save_path = os.path.join(raw_folder, channel_name)
                    io.imsave(save_path, comp_data[j, :, :, idx], check_contrast=False)


def create_tiled_comparison(input_dir_list, output_dir, img_sub_folder='normalized'):
    """Creates a tiled image comparing FOVs from all supplied runs for each channel.

    Args:
        input_dir_list: list of directories to compare
        output_dir: directory where tifs will be saved"""

    # load images
    dir_dict = {}
    dir_shapes = []
    for dir_name in input_dir_list:
        dir_images = load_imgs_from_tree(dir_name, img_sub_folder=img_sub_folder)
        dir_shapes.append(dir_images.shape)
        dir_dict[dir_name] = dir_images

    if not np.all([shape == dir_shapes[0] for shape in dir_shapes]):
        raise ValueError("All directories must contain the same number of fovs and images")

    first_dir = dir_dict[input_dir_list[0]]
    img_size = first_dir.shape[1]
    fov_num = first_dir.shape[0]

    # loop over each channel
    for j in range(first_dir.shape[3]):
        # create tiled array of dirs x fovs
        tiled_image = np.zeros((img_size * len(input_dir_list),
                                img_size * fov_num), dtype=first_dir.dtype)

        # loop over each fov, and place into columns of tiled array
        for i in range(first_dir.shape[0]):
            start = i * img_size
            end = (i + 1) * img_size

            # go through each of the directories and place in appropriate spot
            for idx, key in enumerate(input_dir_list):
                tiled_image[(img_size * idx):(img_size * (idx + 1)), start:end] = \
                    dir_dict[key].values[i, :, :, j]

        io.imsave(os.path.join(output_dir, first_dir.channels.values[j] + '_comparison.tiff'),
                  tiled_image, check_contrast=False)


def add_source_channel_to_tiled_image(raw_img_dir, tiled_img_dir, output_dir, source_channel,
                                      percent_norm=98):
    """Adds the specified source_channel to the first row of previously generated tiled images

    Args:
        raw_img_dir (str): path to directory containing the raw images
        tiled_img_dir (str): path to directory contained the tiled images
        output_dir (str): path to directory where outputs will be saved
        source_channel (str): the channel which will be prepended to the tiled images
        percent_norm (int): percentile normalization param to enable easy visualization"""

    # load source images
    source_imgs = load_imgs_from_tree(raw_img_dir, channels=[source_channel],
                                      dtype='float32')

    # convert stacked images to concatenated row
    source_list = [source_imgs.values[fov, :, :, 0] for fov in range(source_imgs.shape[0])]
    source_row = np.concatenate(source_list, axis=1)
    perc_source = np.percentile(source_row, percent_norm)

    # confirm tiled images have expected shape
    tiled_images = list_files(tiled_img_dir)
    test_file = io.imread(os.path.join(tiled_img_dir, tiled_images[0]))
    if test_file.shape[1] != source_row.shape[1]:
        raise ValueError('Tiled image {} has shape {}, but source image {} has'
                         'shape {}'.format(tiled_images[0], test_file.shape, source_channel,
                                           source_row.shape))

    # loop through each tiled image, prepend source row, and save
    for tile_name in tiled_images:
        current_tile = io.imread(os.path.join(tiled_img_dir, tile_name))

        # normalize the source row to be in the same range as the current tile
        perc_tile = np.percentile(current_tile, percent_norm)
        perc_ratio = perc_source / perc_tile
        normalized_source = source_row / perc_ratio

        # combine together and save
        combined_tile = np.concatenate([normalized_source, current_tile])
        save_name = tile_name.split('.tiff')[0] + '_source_' + source_channel + '.tiff'
        io.imsave(os.path.join(output_dir, save_name), combined_tile, check_contrast=False)


def replace_with_intensity_image(run_dir, channel='Au', replace=True, fovs=None):
    """Replaces the specified channel with the intensity image of that channel

    Args:
        run_dir (str): directory containing extracted run data
        channel (str): the channel whose intensity image will be copied over
        fovs (list or None): the subset of fovs within run_dir which will have their
            intensity image copied over. If None, applies to all fovs
        replace (bool): controls whether intensity image is copied over with _intensity appended
            or if it will overwrite existing channel"""

    all_fovs = list_folders(run_dir)

    # ensure supplied folders are valid
    if fovs is not None:
        verify_in_list(specified_folders=fovs, all_folders=all_fovs)
        all_fovs = fovs

    # ensure channel is valid
    test_file = os.path.join(run_dir, all_fovs[0], 'intensities', channel + '_intensity.tiff')
    if not os.path.exists(test_file):
        raise FileNotFoundError('Could not find specified file {}'.format(test_file))

    # loop through each fov
    for fov in all_fovs:
        # control whether intensity image overwrites previous image or is copied with new name
        if replace:
            suffix = '.tiff'
        else:
            suffix = '_intensity.tiff'
        shutil.copy(os.path.join(run_dir, fov, 'intensities', channel + '_intensity.tiff'),
                    os.path.join(run_dir, fov, channel + suffix))


def create_rosetta_matrices(default_matrix, save_dir, multipliers, masses=None):
    """Creates a series of compensation matrices for evaluating coefficients
    Args:
        default_matrix (str): path to the rosetta matrix to use as the default
        save_dir (str): output directory
        multipliers (list): the range of values to multiply the default matrix by
            to get new coefficients
        masses (list | None): an optional list of masses to include in the multiplication. If
            only a subset of masses are specified, other masses will retain their values
            in all iterations. If None, all masses are included
    """
    # Read input matrix
    comp_matrix = pd.read_csv(default_matrix, index_col=0)
    comp_masses = comp_matrix.index

    # Check channel input
    if masses is None:
        masses = comp_masses
    else:
        try:
            masses = [int(x) for x in masses]
        except ValueError:
            raise ValueError("Masses must be provided as integers")
        verify_in_list(specified_masses=masses, rosetta_masses=comp_masses)

    # loop over each specified multiplier and create separate compensation matrix
    for i in multipliers:
        mult_matrix = copy.deepcopy(comp_matrix)

        for j in range(len(comp_matrix)):
            # multiply specified channel by multiplier
            if comp_masses[j] in masses:
                mult_matrix.iloc[j, :] = comp_matrix.iloc[j, :] * i
        base_name = os.path.basename(default_matrix).split('.csv')[0]
        mult_matrix.to_csv(os.path.join(save_dir, base_name + '_mult_%s.csv' % (str(i))))
