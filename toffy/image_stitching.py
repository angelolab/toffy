import math
import os
import re

import natsort as ns
import skimage.io as io
from tmi import data_utils, image_utils, io_utils, load_utils, misc_utils

from toffy import json_utils


def get_max_img_size(tiff_out_dir, img_sub_folder='', run_dir=None, fov_list=None):
    """Retrieves the maximum FOV image size listed in the run file, or for the given FOVs
        Args:
            tiff_out_dir (str): path to the extracted images for the specific run
            img_sub_folder (str): optional name of image sub-folder within each fov
            run_dir (str): path to the run directory containing the run json files, default None
            fov_list (list): list of fovs to check max size for, default none which check all fovs
        Returns:
            value of max image size"""

    if run_dir:
        run_name = os.path.basename(run_dir)
        run_file_path = os.path.join(run_dir, run_name + '.json')

    img_sizes = []

    # check for a run file
    if run_dir and os.path.exists(run_file_path):
        # retrieve all pixel width dimensions of the fovs
        run_data = json_utils.read_json_file(run_file_path)

        if not fov_list:
            for fov in run_data['fovs']:
                img_sizes.append(fov.get('frameSizePixels')['width'])
        else:
            for fov in fov_list:
                fov_digits = re.findall(r'\d+', fov)
                run = run_data.get('fovs')
                # get data for fov in list
                fov_data = list(filter(lambda fov: fov['runOrder'] == int(fov_digits[0]) and
                                       fov['scanCount'] == int(fov_digits[1]), run))
                img_sizes.append(fov_data[0].get('frameSizePixels')['width'])

    # use extracted images to get max size
    else:
        if not fov_list:
            fov_list = io_utils.list_folders(tiff_out_dir, substrs='fov-')
        channels = io_utils.list_files(os.path.join(tiff_out_dir, fov_list[0], img_sub_folder))
        # check image size for each fov
        for fov in fov_list:
            test_file = io.imread(os.path.join(tiff_out_dir, fov, img_sub_folder, channels[0]))
            img_sizes.append(test_file.shape[1])

    # largest in run
    max_img_size = max(img_sizes)
    return max_img_size


def get_tiled_names(fov_list, run_dir):
    """Retrieves the original tiled name for each fov
        Args:
            fov_list (list): list of fovs that have an existing image dir
            run_dir (str): path to the run directory containing the run json file
        Returns:
            dictionary with RnCm name as keys and the fov-x-scan-1 name as values"""

    run_name = os.path.basename(run_dir)
    run_file_path = os.path.join(run_dir, run_name + '.json')
    fov_names = {}

    # check for a run file
    io_utils.validate_paths(run_file_path)

    # retrieve all tiled fov names
    run_data = json_utils.read_json_file(run_file_path)
    for fov in run_data['fovs']:
        run_order = fov.get('runOrder')
        default_name = f'fov-{run_order}-scan-1'

        if default_name in fov_list:
            # get tiled name
            tiled_name = fov.get('name')
            # filter out moly fovs
            if tiled_name != 'MoQC':
                fov_names[tiled_name] = default_name

    return fov_names


def stitch_images(tiff_out_dir, run_dir=None, channels=None, img_sub_folder=None, tiled=False):
    """Creates a new directory containing stitched channel images for the run
        Args:
            tiff_out_dir (str): path to the extracted images for the specific run
            run_dir (str): path to the run directory containing the run json files, default None
            channels (list): list of channels to produce stitched images for, None will do all
            img_sub_folder (str): optional name of image sub-folder within each fov
            tiled (bool): whether to stitch images back into original tiled shape """

    io_utils.validate_paths(tiff_out_dir)
    if run_dir:
        io_utils.validate_paths(run_dir)

    # check for previous stitching
    stitched_dir = os.path.join(tiff_out_dir, 'stitched_images')
    if tiled:
        stitched_dir = os.path.join(tiff_out_dir, 'stitched_images_tiled')
        if run_dir is None:
            raise ValueError("You must provide the run directory to stitch images into their "
                             "original tiled shape.")
    if os.path.exists(stitched_dir):
        raise ValueError(f"The stitch_images subdirectory already exists in {tiff_out_dir}")

    folders = io_utils.list_folders(tiff_out_dir, substrs='fov-')
    folders = ns.natsorted(folders)

    # no img_sub_folder, change to empty string to read directly from base folder
    if img_sub_folder is None:
        img_sub_folder = ""

    # retrieve all extracted channel names, or verify the list provided
    if channels is None:
        channels = io_utils.remove_file_extensions(
            io_utils.list_files(
                dir_name=os.path.join(tiff_out_dir, folders[0], img_sub_folder),
                substrs='.tiff')
            )
    else:
        misc_utils.verify_in_list(
            channel_inputs=channels,
            valid_channels=io_utils.remove_file_extensions(
                io_utils.list_files(dir_name=os.path.join(
                    tiff_out_dir, folders[0], img_sub_folder),
                                    substrs='.tiff')
                )
            )

    # get load and stitching args
    if tiled:
        folders_dict = get_tiled_names(folders, run_dir)
        # returns a dict with keys RnCm and values og folder names
        try:
            expected_fovs, num_rows, num_cols = load_utils.get_tiled_fov_names(
                list(folders_dict.keys()), return_dims=True)
        except AttributeError:
            raise ValueError(f'FOV names found in the run file were not in tiled (RnCm) format.')
    else:
        num_cols = math.isqrt(len(folders))
        max_img_size = get_max_img_size(tiff_out_dir, img_sub_folder, run_dir)

    # make stitched subdir
    os.makedirs(stitched_dir)

    # save the stitched images to the stitched_image subdir
    for chan in channels:
        # tiled image loading
        if tiled:
            image_data = load_utils.load_tiled_img_data(tiff_out_dir, folders_dict, expected_fovs,
                                                        chan, single_dir=False,
                                                        img_sub_folder=img_sub_folder)
        else:
            image_data = load_utils.load_imgs_from_tree(tiff_out_dir,
                                                        img_sub_folder=img_sub_folder,
                                                        fovs=folders, channels=[chan],
                                                        max_image_size=max_img_size)
        stitched = data_utils.stitch_images(image_data, num_cols)
        current_img = stitched.loc['stitched_image', :, :, chan].values
        fname = os.path.join(stitched_dir, chan + "_stitched.tiff")
        image_utils.save_image(fname, current_img)
