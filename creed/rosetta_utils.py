import os
import json

import numpy as np
import pandas as pd

import skimage.io as io
from scipy.ndimage import gaussian_filter

from ark.utils.load_utils import load_imgs_from_tree, load_imgs_from_dir
from ark.utils.io_utils import list_folders
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
        chan_coeffs = \
            np.expand_dims(np.expand_dims(np.expand_dims(chan_coeffs, axis=0), axis=0), axis=0)

        # broadcast across entire dataset and collapse into single set of values
        chan_vals = raw_inputs * chan_coeffs
        chan_vals = np.sum(chan_vals, axis=-1)

        # subtract compensated values from target channel
        outputs[..., chan] -= chan_vals

    return outputs


def compensate_image_data(raw_data_dir, comp_data_dir, comp_mat_path, panel_info_path,
                          save_format='normalized', batch_size=10, gaus_rad=1):
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
    """

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
    save_format = save_format.lower()
    verify_in_list(save_format=save_format, allowed_formats=allowed_formats)

    if batch_size < 1 or not batch_size.dtype == 'int':
        raise ValueError('batch_size parameter must be a positive integer')

    if gaus_rad < 0 or not gaus_rad.dtype == 'int':
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
            batch_data_blurred = np.zeros_like(batch_data)
            for j in range(batch_data.shape[0]):
                for k in range(batch_data.shape[-1]):
                    blurred = gaussian_filter(batch_data[j, :, :, k], sigma=gaus_rad)
                    batch_data_blurred[j, :, :, k] = blurred
        else:
            batch_data_blurred = batch_data

        comp_data = compensate_matrix_simple(raw_inputs=batch_data_blurred,
                                             comp_coeffs=comp_mat.values)

        # set negative values to zero
        comp_data = np.where(comp_data > 0, comp_data, 0)

        # save data
        for j in range(batch_data.shape[0]):
            fov_folder = os.path.join(comp_data_dir, batch_data.fovs.values[j])
            os.makedirs(fov_folder)

            # create directories for saving tifs
            if save_format == 'normalized' or save_format == 'both':
                norm_folder = os.path.join(fov_folder, 'normalized')
                os.makedirs(norm_folder)

            if save_format == 'raw' or save_format == 'both':
                raw_folder = os.path.join(fov_folder, 'raw')
                os.makedirs(raw_folder)

            for k in range(batch_data.shape[-1]):
                channel_name = batch_data.channels.values[k] + '.tiff'

                # save tifs to appropriate directories
                if save_format == 'normalized' or save_format == 'both':
                    save_path = os.path.join(norm_folder, channel_name)
                    io.imsave(save_path, comp_data[j, :, :, k] / 100, check_contrast=False)

                if save_format == 'raw' or save_format == 'both':
                    save_path = os.path.join(raw_folder, channel_name)
                    io.imsave(save_path, comp_data[j, :, :, k], check_contrast=False)


def compare_comped_images(raw_dir, comp_dir_list, output_dir):
    """Creates a tiled image containing the raw image, difference image, and output image
    from each compensated directory supplied

    Args:
        raw_dir: directory containing raw images
        comp_dir_list: list of directories containing compensated images
        output_dir: directory where tifs will be saved"""

    # load images
    raw_images = load_imgs_from_tree(raw_dir, dtype='float')
    comp_dict = {}
    for dir_name in comp_dir_list:
        comp_images = load_imgs_from_tree(dir_name, dtype='float')
        comp_dict[dir_name] = comp_images

    img_size = raw_images.shape[1]
    fov_num = raw_images.shape[0]
    # compute difference between first compensation image and baseline
    diff_images = raw_images.values - comp_dict[comp_dir_list[0]].values
    diff_images[diff_images < 0] = 0

    comp_image_num = len(comp_dir_list)
    # loop over each channel
    for j in range(raw_images.shape[3]):
        # create tiled array of corrections x fovs
        tiled_image = np.zeros((img_size * (comp_image_num + 2),
                                img_size * fov_num))
        # loop over each fov, and place into columns of tiled array
        for i in range(raw_images.shape[0]):
            start = i * img_size
            end = (i + 1) * img_size

            # first row is raw image
            tiled_image[:img_size, start:end] = raw_images.values[i, :, :, j]

            # second row is difference image between raw and first compensated image
            tiled_image[img_size:(img_size * 2), start:end] = diff_images[i, :, :, j]

            # subsequent rows are compensated images with different coefficients
            for idx, key in enumerate(comp_dir_list):
                tiled_image[(img_size * (idx + 2)):(img_size * (idx + 3)), start:end] = \
                    comp_dict[key].values[i, :, :, j]

        io.imsave(os.path.join(output_dir, raw_images.channels.values[j] + '_comparison.tiff'), tiled_image)


def compare_paired_runs(input_dir_list, output_dir):
    """Creates a tiled image containing paired images from multiple runs

    Args:
        input_dir_list: list of directories to compare
        output_dir: directory where tifs will be saved"""

    # load images
    comp_dict = {}
    for dir_name in input_dir_list:
        comp_images = load_imgs_from_tree(dir_name, dtype='float')
        comp_dict[dir_name] = comp_images

    first_dir = comp_dict[input_dir_list[0]]
    img_size = first_dir.shape[1]
    fov_num = first_dir.shape[0]

    # loop over each channel
    for j in range(first_dir.shape[3]):
        # create tiled array of dirs x fovs
        tiled_image = np.zeros((img_size * len(input_dir_list),
                                img_size * fov_num))

        # loop over each fov, and place into columns of tiled array
        for i in range(first_dir.shape[0]):
            start = i * img_size
            end = (i + 1) * img_size

            # go through each of the directories and place in appropriate spot
            for idx, key in enumerate(input_dir_list):
                tiled_image[(img_size * idx):(img_size * (idx + 1)), start:end] = \
                    comp_dict[key].values[i, :, :, j]

        io.imsave(os.path.join(output_dir, first_dir.channels.values[j] + '_comparison.tiff'),
                  tiled_image / 200)
