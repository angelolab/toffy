import os
import json

import numpy as np
import pandas as pd

import skimage.io as io
from scipy.ndimage import gaussian_filter

from ark.utils.load_utils import load_imgs_from_tree, load_imgs_from_dir
from ark.utils.io_utils import list_folders, validate_paths
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


def compensate_matrix_simple(raw_inputs, comp_coeffs):
    """Perform compensation on the raw data using the supplied compensation values

    Args:
        raw_inputs (numpy.ndarray):
            array with shape [fovs, rows, cols, channels] containing the image data
        comp_coeffs (numpy.ndarray):
            2D array of coefficients with source channels as rows and targets as columns

    returns:
        numpy.ndarray: compensated copy of the raw inputs"""

    outputs = np.copy(raw_inputs)

    # loop over each channel and construct compensation values
    for chan in range(raw_inputs.shape[-1]):
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

    return outputs


def compensate_image_data(raw_data_dir, comp_data_dir, comp_mat_path, panel_info_path,
                          save_format='normalized', batch_size=10, gaus_rad=1, norm_const=100):
    """Function to compensate MIBI data with a flow-cytometry style compensation matrix

    Args:
        raw_data_dir: path to directory containing raw images
        comp_data_dir: path to directory where compensated images will be saved
        comp_mat_path: path to compensation matrix, nxn with channel labels
        panel_info_path: path to panel info file with 'masses' and 'targets' as columns
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

    validate_paths([raw_data_dir, comp_data_dir, comp_mat_path, panel_info_path],
                   data_prefix=False)

    # get list of all fovs
    fovs = list_folders(raw_data_dir, substrs=['fov'])

    # load csv files
    comp_mat = pd.read_csv(comp_mat_path, index_col=0)
    panel_info = pd.read_csv(panel_info_path)
    acquired_masses = panel_info['masses'].values
    acquired_targets = panel_info['targets'].values
    all_masses = comp_mat.columns.values.astype('int')

    # make sure panel is in increasing order
    if not np.all(acquired_masses == sorted(acquired_masses)):
        raise ValueError("Masses must be sorted numerically in the panel_info file")

    # make sure channels in comp matrix are same as those in panel csv
    verify_same_elements(acquired_masses=acquired_masses, compensation_masses=all_masses)

    # check first FOV to make sure all channels are present
    test_data = load_imgs_from_tree(data_dir=raw_data_dir, fovs=fovs[0:1],
                                    channels=acquired_targets, dtype='float32')

    verify_same_elements(image_files=test_data.channels.values, listed_channels=acquired_targets)

    # check for valid save_formats
    allowed_formats = ['raw', 'normalized', 'both']
    verify_in_list(save_format=save_format, allowed_formats=allowed_formats)

    if batch_size < 1 or not isinstance(batch_size, int):
        raise ValueError('batch_size parameter must be a positive integer')

    if gaus_rad < 0 or not isinstance(gaus_rad, int):
        raise ValueError('gaus_rad parameter must be a non-negative integer')

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

        comp_data = compensate_matrix_simple(raw_inputs=batch_data,
                                             comp_coeffs=comp_mat.values)

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

            for k in range(batch_data.shape[-1]):
                channel_name = batch_data.channels.values[k] + '.tiff'

                # save tifs to appropriate directories
                if save_format in ['normalized', 'both']:
                    save_path = os.path.join(norm_folder, channel_name)
                    io.imsave(save_path, comp_data[j, :, :, k] / norm_const, check_contrast=False)

                if save_format in ['raw', 'both']:
                    save_path = os.path.join(raw_folder, channel_name)
                    io.imsave(save_path, comp_data[j, :, :, k], check_contrast=False)


def create_tiled_comparison(input_dir_list, output_dir):
    """Creates a tiled image comparing FOVs from all supplied runs for each channel.

    Args:
        input_dir_list: list of directories to compare
        output_dir: directory where tifs will be saved"""

    # load images
    dir_dict = {}
    dir_shapes = []
    for dir_name in input_dir_list:
        dir_images = load_imgs_from_tree(dir_name)
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
                  tiled_image)


def create_rosetta_matrices(default_matrix, multipliers=[0.5, 1, 1.5], channels=None):
    """Creates a series of compensation matrices for evaluating coefficients

    Args:
        default_matrix (str): path to the rosetta matrix to use as the default
        multipliers (list): the range of values to multiply the default matrix by
            to get new coefficients
        channels (list | None): an optional list of channels to include in the multiplication. If
            only a subset of channels are specified, other channels will retain their values
            in all iterations. If None, all channels are included"""

    # TODO: Cameron will make this function

    # step 1: read in the default matrix

    # step 2: figure out which channels will be modified

    # step 3: loop over each of the multipliers

    # step 4: modify the appropriate channels based on mulitiplier

    # step 5: save the modified matrix as original_name_multiplier

    pass
