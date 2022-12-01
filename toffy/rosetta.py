import copy
import json
import os
import random
import shutil

import natsort as ns
import numpy as np
import pandas as pd
import skimage.io as io
from scipy.ndimage import gaussian_filter
from tmi import image_utils, io_utils, load_utils, misc_utils

from toffy.json_utils import read_json_file
from toffy.streak_detection import streak_correction


def transform_compensation_json(json_path, comp_mat_path):
    """Converts the JSON file from ionpath into a compensation matrix

    Args:
        json_path (str): path to json file
        comp_mat_path (str): path to comp matrix

    returns:
        pd.DataTable: matrix with sources channels as rows and target channels as columns"""

    data = read_json_file(json_path)['Data']

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
                    output_masses, all_masses, fovs, save_format, raw_data_sub_folder, batch_size,
                    gaus_rad):
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
        raw_data_sub_folder (string): sub-folder for raw images
        batch_size (int): number of images to process concurrently
        gaus_rad (int): radius for smoothing"""

    # make sure panel is in increasing order
    if not np.all(acquired_masses == sorted(acquired_masses)):
        raise ValueError("Masses must be sorted numerically in the panel_info file")

    # make sure channels in comp matrix are same as those in panel csv
    misc_utils.verify_same_elements(acquired_masses=acquired_masses,
                                    compensation_masses=all_masses)

    # check first FOV to make sure all channels are present
    test_data = load_utils.load_imgs_from_tree(data_dir=raw_data_dir, fovs=fovs[0:1],
                                               img_sub_folder=raw_data_sub_folder)

    misc_utils.verify_in_list(listed_channels=acquired_targets,
                              image_files=test_data.channels.values)

    # make sure supplied masses are present
    if input_masses is not None:
        misc_utils.verify_in_list(input_masses=input_masses,
                                  compensation_masses=all_masses)

    if output_masses is not None:
        misc_utils.verify_in_list(output_masses=output_masses,
                                  compensation_masses=all_masses)

    # make sure compensation matrix has valid values
    if comp_mat.isna().values.any():
        raise ValueError('Compensation matrix must contain a value for every field; check to '
                         'make sure there are no missing values')

    # check for valid save_formats
    allowed_formats = ['raw', 'rescaled', 'both']
    misc_utils.verify_in_list(save_format=save_format, allowed_formats=allowed_formats)

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size parameter must be a positive integer')

    if not isinstance(gaus_rad, int) or gaus_rad < 0:
        raise ValueError('gaus_rad parameter must be a non-negative integer')


def clean_rosetta_test_dir(folder_path):
    """Remove the unnecessary intermediate folders created by rosetta test data computation

    Args:
        folder_path (str): base dir for testing, image subdirs will be stored here
    """

    # remove the compensated data folders
    comp_folders = io_utils.list_folders(folder_path, substrs='compensated_data_')
    for cf in comp_folders:
        shutil.rmtree(os.path.join(folder_path, cf))

    # remove the stitched image folder
    shutil.rmtree(os.path.join(folder_path, 'stitched_images'))


def flat_field_correction(img, gaus_rad=100):
    """Apply flat field correction to an image

    Args:
        img (np.ndarray): image to be corrected
        gaus_rad (int): radius for smoothing

    Returns:
        np.ndarray: corrected image """

    # smooth image
    img_smooth = gaussian_filter(img, sigma=gaus_rad)

    # calculate mean to preserve overall intensity
    img_mean = np.mean(img)

    # apply correction
    img_corr = (img / img_smooth) * img_mean

    return img_corr


def get_masses_from_channel_names(names, panel_df):
    """Get the weights for the given names from the input dataframe.

    Args:
        names (list): the channels whose masses will be returned
        panel_df (pd.DataFrame): the panel containing the masses and channel names

    Returns:
        list: the masses for the given channels
    """

    misc_utils.verify_in_list(supplied_channel_names=names,
                              panel_channel_names=panel_df['Target'].values)

    weights = panel_df.loc[np.isin(panel_df['Target'], names)]['Mass'].values

    return weights


def compensate_image_data(raw_data_dir, comp_data_dir, comp_mat_path, panel_info,
                          input_masses=None, output_masses=None, save_format='rescaled',
                          raw_data_sub_folder='', batch_size=1, gaus_rad=1, norm_const=200,
                          ffc_masses=[39], correct_streaks=False, streak_chan='Noodle'):
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
            'rescaled': all images are divided by 200 to enable visualization. This transform
                has no effect on downstream analysis as it preserves relative expression values
            'both': saves both 'raw' and 'rescaled' images
        raw_data_sub_folder (string): sub-folder for raw images
        batch_size: number of images to process at a time
        gaus_rad: radius for blurring image data. Passing 0 will result in no blurring
        norm_const: constant used for rescaling if save_format == 'rescaled'
        ffc_masses (list): masses that need to be flat field corrected.
        correct_streaks (bool): whether to correct streaks in the image
        streak_chan (str): the channel to use for streak correction
    """

    io_utils.validate_paths([raw_data_dir, comp_data_dir, comp_mat_path])

    # get list of all fovs
    fovs = io_utils.list_folders(raw_data_dir, substrs='fov')

    # load csv files
    comp_mat = pd.read_csv(comp_mat_path, index_col=0)
    acquired_masses = panel_info['Mass'].values
    acquired_targets = panel_info['Target'].values
    all_masses = comp_mat.columns.values.astype('int')

    # convert ffc mass into ffc channel names
    ffc_channels = [panel_info.loc[panel_info.Mass == mass].Target.values[0] for
                    mass in ffc_masses]

    validate_inputs(raw_data_dir, comp_mat, acquired_masses, acquired_targets, input_masses,
                    output_masses, all_masses, fovs, save_format, raw_data_sub_folder, batch_size,
                    gaus_rad)

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
        print("Processing image {}".format(i+1))

        # load batch of fovs
        batch_fovs = fovs[i: i + batch_size]
        batch_data = load_utils.load_imgs_from_tree(data_dir=raw_data_dir, fovs=batch_fovs,
                                                    channels=acquired_targets,
                                                    img_sub_folder=raw_data_sub_folder)

        # convert to float32 for gaussian_filter and rosetta compatibility
        batch_data = batch_data.astype(np.float32)

        # blur data
        if gaus_rad > 0:
            for j in range(batch_data.shape[0]):
                for k in range(batch_data.shape[-1]):
                    batch_data[j, :, :, k] = gaussian_filter(batch_data[j, :, :, k],
                                                             sigma=gaus_rad)

        if correct_streaks:
            corrected_channels, _ = streak_correction(fov_data=batch_data,
                                                      streak_channel=streak_chan)
            batch_data[0] = corrected_channels.values

        # apply flat field correction if specified
        if ffc_channels is not None:
            misc_utils.verify_in_list(flat_field_correction_masses=ffc_channels,
                                      all_masses=acquired_targets)
            for fov in batch_fovs:
                for chan in ffc_channels:
                    raw_img = batch_data.loc[fov, :, :, chan].values
                    batch_data.loc[fov, :, :, chan] = flat_field_correction(raw_img)

        comp_data = _compensate_matrix_simple(raw_inputs=batch_data,
                                              comp_coeffs=comp_mat.values,
                                              out_indices=out_indices)

        # save data
        for j in range(batch_data.shape[0]):
            fov_folder = os.path.join(comp_data_dir, batch_data.fovs.values[j])
            os.makedirs(fov_folder)

            # create directories for saving tifs
            if save_format in ['rescaled', 'both']:
                rescale_folder = os.path.join(fov_folder, 'rescaled')
                os.makedirs(rescale_folder)

            if save_format in ['raw', 'both']:
                raw_folder = os.path.join(fov_folder, 'raw')
                os.makedirs(raw_folder)

            # this may be only a subset of masses, based on output_masses
            for idx, val in enumerate(out_indices):
                channel_name = batch_data.channels.values[val] + '.tiff'

                # save tifs to appropriate directories
                if save_format in ['rescaled', 'both']:
                    save_path = os.path.join(rescale_folder, channel_name)
                    image_utils.save_image(save_path, comp_data[j, :, :, idx] / norm_const)

                if save_format in ['raw', 'both']:
                    save_path = os.path.join(raw_folder, channel_name)
                    image_utils.save_image(save_path, comp_data[j, :, :, idx])


def create_tiled_comparison(input_dir_list, output_dir, max_img_size,
                            img_sub_folder='rescaled', channels=None):
    """Creates a tiled image comparing FOVs from all supplied runs for each channel.

    Args:
        input_dir_list: list of directories to compare
        output_dir: directory where tifs will be saved
        max_img_size (int): largest fov image size
        img_sub_folder: subfolder within each input directory to load images from
        channels: list of channels to compare. """

    test_dir = input_dir_list[0]
    test_fov = io_utils.list_folders(test_dir)[0]
    test_data = load_utils.load_imgs_from_tree(data_dir=test_dir, fovs=[test_fov],
                                               img_sub_folder=img_sub_folder, channels=channels)

    channels = test_data.channels.values
    chanel_num = len(channels)

    # check that all dirs have the same number of fovs and correct subset of channels
    fov_names = io_utils.list_folders(input_dir_list[0])
    for dir_name in input_dir_list[1:]:
        current_folders = io_utils.list_folders(dir_name)
        misc_utils.verify_same_elements(fov_names1=fov_names, fov_names2=current_folders)
        current_channels = load_utils.load_imgs_from_tree(data_dir=dir_name,
                                                          img_sub_folder=img_sub_folder,
                                                          fovs=current_folders[:1]).channels.values
        misc_utils.verify_in_list(specified_channels=channels, current_channels=current_channels)

    fov_num = len(fov_names)

    # loop over each channel
    for j in range(chanel_num):
        # create tiled array of dirs x fovs
        tiled_image = np.zeros((max_img_size * len(input_dir_list),
                                max_img_size * fov_num), dtype=test_data.dtype)

        # loop over each fov, and place into columns of tiled array
        for i in range(fov_num):
            start = i * max_img_size
            end = (i + 1) * max_img_size

            # go through each of the directories, read in the images, and place in the right spot
            for idx, key in enumerate(input_dir_list):
                dir_data = load_utils.load_imgs_from_tree(key, channels=channels[j:j + 1],
                                                          img_sub_folder=img_sub_folder,
                                                          max_image_size=max_img_size)
                tiled_image[(max_img_size * idx):(max_img_size * (idx + 1)), start:end] = \
                    dir_data.values[i, :, :, 0]
        fname = os.path.join(output_dir, channels[j] + "_comparison.tiff")
        image_utils.save_image(fname, tiled_image)


def add_source_channel_to_tiled_image(raw_img_dir, tiled_img_dir, output_dir, source_channel,
                                      max_img_size, img_sub_folder='', percent_norm=98):
    """Adds the specified source_channel to the first row of previously generated tiled images

    Args:
        raw_img_dir (str): path to directory containing the raw images
        tiled_img_dir (str): path to directory contained the tiled images
        output_dir (str): path to directory where outputs will be saved
        img_sub_folder (str): subfolder within raw_img_dir to load images from
        max_img_size (int): largest fov image size
        source_channel (str): the channel which will be prepended to the tiled images
        percent_norm (int): percentile normalization param to enable easy visualization"""

    # load source images
    source_imgs = load_utils.load_imgs_from_tree(raw_img_dir, channels=[source_channel],
                                                 img_sub_folder=img_sub_folder,
                                                 max_image_size=max_img_size)

    # convert stacked images to concatenated row
    source_list = [source_imgs.values[fov, :, :, 0] for fov in range(source_imgs.shape[0])]
    source_row = np.concatenate(source_list, axis=1)
    perc_source = np.percentile(source_row, percent_norm)

    # confirm tiled images have expected shape
    tiled_images = io_utils.list_files(tiled_img_dir)
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
        rescaled_source = source_row / perc_ratio

        # combine together and save
        combined_tile = np.concatenate([rescaled_source, current_tile])
        save_name = tile_name.split('.tiff')[0] + '_source_' + source_channel + '.tiff'
        image_utils.save_image(os.path.join(output_dir, save_name), combined_tile)


def replace_with_intensity_image(run_dir, channel='Au', replace=True, fovs=None):
    """Replaces the specified channel with the intensity image of that channel

    Args:
        run_dir (str): directory containing extracted run data
        channel (str): the channel whose intensity image will be copied over
        fovs (list or None): the subset of fovs within run_dir which will have their
            intensity image copied over. If None, applies to all fovs
        replace (bool): controls whether intensity image is copied over with _intensity appended
            or if it will overwrite existing channel"""

    all_fovs = io_utils.list_folders(run_dir)

    # ensure supplied folders are valid
    if fovs is not None:
        misc_utils.verify_in_list(specified_folders=fovs, all_folders=all_fovs)
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


def remove_sub_dirs(run_dir, sub_dirs, fovs=None):
    """Removes specified sub-folders from fovs in a run

    Args:
        run_dir (str): path to directory containing fovs
        sub_dirs (list): directories to remove from each fov
        fovs (list): list of fovs to remove dirs from, otherwise removes from all fovs
    """

    all_fovs = io_utils.list_folders(run_dir)

    # ensure supplied folders are valid
    if fovs is not None:
        misc_utils.verify_in_list(specified_folders=fovs, all_folders=all_fovs)
        all_fovs = fovs

    # ensure all sub_dirs exist
    for sub_dir in sub_dirs:
        if not os.path.isdir(os.path.join(run_dir, all_fovs[0], sub_dir)):
            raise ValueError("Did not find {} in {}".format(sub_dir, all_fovs[0]))

    for fov in all_fovs:
        for sub_dir in sub_dirs:
            shutil.rmtree(os.path.join(run_dir, fov, sub_dir))


def create_rosetta_matrices(default_matrix, save_dir, multipliers, current_channel_name,
                            output_channel_names, masses=None):
    """Creates a series of compensation matrices for evaluating coefficients
    Args:
        default_matrix (str): path to the rosetta matrix to use as the default
        save_dir (str): output directory
        multipliers (list): the range of values to multiply the default matrix by
            to get new coefficients
        current_channel_name (str): channel being adjusted
        output_channel_names (list): subset of the channels to compensate for
        masses (list | None): an optional list of masses to include in the multiplication. If
            only a subset of masses are specified, other masses will retain their values
            in all iterations. If None, all masses are included
    """
    # Read input matrix
    comp_matrix = pd.read_csv(default_matrix, index_col=0)
    comp_masses = comp_matrix.index

    # Check that all entries of comp_matrix are numeric
    if not np.issubdtype(comp_matrix.values.dtype, np.number):
        raise ValueError('Compensation matrix must include only numeric entries')

    # Check channel input
    if masses is None:
        masses = comp_masses
    else:
        try:
            masses = [int(x) for x in masses]
        except ValueError:
            raise ValueError("Masses must be provided as integers")
        misc_utils.verify_in_list(specified_masses=masses, rosetta_masses=comp_masses)

    # define the file prefix used for each compensation matrix file
    default_mat_prefix = os.path.basename(default_matrix).split('.csv')[0]
    output_chan_str = '_'.join(output_channel_names) if output_channel_names is not None else 'all'
    comp_file_prefix = f'{current_channel_name}_{output_chan_str}_{default_mat_prefix}_mult'

    # loop over each specified multiplier and create separate compensation matrix
    for i in multipliers:
        mult_matrix = copy.deepcopy(comp_matrix)

        for j in range(len(comp_matrix)):
            # multiply specified channel by multiplier
            if comp_masses[j] in masses:
                mult_matrix.iloc[j, :] = comp_matrix.iloc[j, :] * i
        comp_name = f'{comp_file_prefix}_{i}.csv'
        mult_matrix.to_csv(os.path.join(save_dir, comp_name))


def copy_image_files(cohort_name, run_names, rosetta_testing_dir, extracted_imgs_dir,
                     fovs_per_run=5):
    """ Creates a new directory for rosetta testing and copies over a random subset of
        previously extracted images
    Args:
        cohort_name (str): name for all combined runs
        run_names (list): gives names of run folders to retrieve extracted images from
        rosetta_testing_dir (str): directory where to create cohort rosetta testing folder
        extracted_imgs_dir (str): directory containing images from each run,
        fovs_per_run: number of fovs from each run to use for testing, default 5

    """
    # path validation
    io_utils.validate_paths([rosetta_testing_dir, extracted_imgs_dir])

    # validate provided run names
    small_runs = []
    for run in run_names:
        if not os.path.exists(os.path.join(extracted_imgs_dir, run)):
            raise ValueError(f'{run} is not a valid run name found in {extracted_imgs_dir}')
        fovs_in_run = io_utils.list_folders(os.path.join(extracted_imgs_dir, run), substrs='fov')
        # check number of fovs in each run
        if len(fovs_in_run) < fovs_per_run:
            small_runs.append(run)
    if len(small_runs) > 0:
        raise ValueError(f"The run folders {small_runs} do not contain the minimum amount of FOVs "
                         f"({fovs_per_run}) defined by the fovs_per_run given.")

    # make rosetta testing dir and extracted images subdir
    cohort_rosetta_dir = os.path.join(rosetta_testing_dir, cohort_name)
    os.makedirs(os.path.join(cohort_rosetta_dir, 'extracted_images'))

    # randomly choose fovs from a run and copy them to the img subdir in rosetta testing dir
    for i, run in enumerate(ns.natsorted(run_names)):
        run_path = os.path.join(extracted_imgs_dir, run)

        fovs_in_run = io_utils.list_folders(run_path, substrs='fov')
        fovs_in_run = ns.natsorted(fovs_in_run)
        rosetta_fovs = random.sample(fovs_in_run, k=fovs_per_run)

        for fov in rosetta_fovs:
            fov_path = os.path.join(os.path.join(extracted_imgs_dir, run, fov))
            # prepend the run name to each fov
            new_path = os.path.join(os.path.join(cohort_rosetta_dir, 'extracted_images',
                                                 run + '_' + fov))
            shutil.copytree(fov_path, new_path)


def rescale_raw_imgs(img_out_dir, scale=200):
    """ Rescale image data to be between 0 and 1
    Args:
        img_out_dir (str): the directory containing extracted images
        scale (int): how much to rescale the image data by

    Returns:
        saves the rescaled images in a subdir
    """
    io_utils.validate_paths(img_out_dir)

    fovs = io_utils.list_folders(img_out_dir)
    for fov in fovs:
        fov_dir = os.path.join(img_out_dir, fov)
        # create subdirectory for the new images
        sub_dir = os.path.join(fov_dir, 'rescaled')
        os.makedirs(sub_dir)
        chans = io_utils.list_files(fov_dir)
        # rescale each channel image
        for chan in chans:
            img = io.imread(os.path.join(fov_dir, chan))
            img = (img / scale).astype('float32')
            fname = os.path.join(sub_dir, chan)
            image_utils.save_image(fname, img)


def generate_rosetta_test_imgs(rosetta_mat_path, img_out_dir,  multipliers, folder_path, panel,
                               current_channel_name='Noodle', output_channel_names=None):
    """ Compensate example FOV images based on given multipliers
    Args:
        rosetta_mat_path (str): path to rosetta compensation matrix
        img_out_dir (str): directory where extracted images are stored
        multipliers (list): list of coeffient multipliers to create different matrices for
        folder_path (str): base dir for testing, image subdirs will be stored here
        panel (pd.DataFrame): the panel containing the masses and channel names
        current_channel_name (str): channel being adjusted, default Noodle
        output_channel_names (list): subset of the channels to compensate for, default None is all

    Returns:
        Create subdirs containing rosetta compensated images for each multiplier and stitched imgs
    """
    io_utils.validate_paths([rosetta_mat_path, img_out_dir, folder_path])

    # get mass information
    current_channel_mass = get_masses_from_channel_names([current_channel_name], panel)

    if output_channel_names is not None:
        output_masses = get_masses_from_channel_names(output_channel_names, panel)
    else:
        output_masses = None

    # generate rosetta matrices for each multiplier
    create_rosetta_matrices(default_matrix=rosetta_mat_path, save_dir=folder_path,
                            multipliers=multipliers, current_channel_name=current_channel_name,
                            output_channel_names=output_channel_names, masses=current_channel_mass)
    matrix_name = io_utils.remove_file_extensions([os.path.basename(rosetta_mat_path)])[0]

    # loop over each multiplier and compensate the data
    rosetta_dirs = [img_out_dir]
    for multiplier in multipliers:
        rosetta_mat_path = os.path.join(folder_path, f'{matrix_name}_mult_{multiplier}.csv')
        rosetta_out_dir = os.path.join(folder_path, 'compensated_data_{}'.format(multiplier))
        rosetta_dirs.append(rosetta_out_dir)
        os.makedirs(rosetta_out_dir)
        compensate_image_data(raw_data_dir=img_out_dir, comp_data_dir=rosetta_out_dir,
                              comp_mat_path=rosetta_mat_path, raw_data_sub_folder='rescaled',
                              panel_info=panel, batch_size=1, norm_const=1,
                              output_masses=output_masses)
