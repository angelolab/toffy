import math
import os
import re
from collections import Counter

import natsort as ns
import numpy as np
import skimage.io as io
import xarray as xr
from alpineer import data_utils, image_utils, io_utils, load_utils, misc_utils
from skimage import transform

from toffy import json_utils


def get_max_img_size(tiff_out_dir, img_sub_folder="", run_dir=None, fov_list=None):
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
        run_file_path = os.path.join(run_dir, run_name + ".json")

    img_sizes = []

    # check for a run file
    if run_dir and os.path.exists(run_file_path):
        # retrieve all pixel width dimensions of the fovs
        run_data = json_utils.read_json_file(run_file_path)

        if not fov_list:
            for fov in run_data["fovs"]:
                img_sizes.append(fov.get("frameSizePixels")["width"])
        else:
            for fov in fov_list:
                fov_digits = re.findall(r"\d+", fov)
                run = run_data.get("fovs")
                # get data for fov in list
                fov_data = list(
                    filter(
                        lambda fov: fov["runOrder"] == int(fov_digits[0])
                        and fov["scanCount"] == int(fov_digits[1]),
                        run,
                    )
                )
                img_sizes.append(fov_data[0].get("frameSizePixels")["width"])

    # use extracted images to get max size
    else:
        if not fov_list:
            fov_list = io_utils.list_folders(tiff_out_dir, substrs="fov-")
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
    run_file_path = os.path.join(run_dir, run_name + ".json")
    fov_names = {}

    # check for a run file
    io_utils.validate_paths(run_file_path)

    # retrieve all tiled fov names
    run_data = json_utils.read_json_file(run_file_path)
    for fov in run_data["fovs"]:
        run_order = fov.get("runOrder")
        default_name = f"fov-{run_order}-scan-1"

        if default_name in fov_list:
            # get tiled name
            tiled_name = fov.get("name")
            # filter out non-tiled and moly fovs
            if re.search(re.compile(r"(R\+?\d+)(C\+?\d+)"), tiled_name) and tiled_name != "MoQC":
                fov_names[tiled_name] = default_name

    return fov_names


def rescale_stitched_array(img_data, scale):
    """Take an array of stitched image data and rescale the rows and cols dims
    Args:
        img_data (xarray.DataArray): data array with 4 dimensions (fov, rows, cols, channels)
        scale (int): amount to scale the data up or down
    Returns:
        xarray.DataArray: data reshaped according to scale value while keeping fov/channel info
    """

    # extract dimension info from xarray
    fov_num, rows, cols, chan_num = img_data.shape
    fovs = img_data.fovs.values
    channels = img_data.channels.values

    # create empty array
    rescaled_data = np.zeros((fov_num, int(rows * scale), int(cols * scale), chan_num))

    # scale the data for each img individually
    for i in range(fov_num):
        for j in range(chan_num):
            current_img = np.array(img_data[i, :, :, j])
            rescaled_data[i, :, :, j] = rescale_image(current_img, scale)

    # fill in the metadata
    rescaled_xr = xr.DataArray(
        rescaled_data,
        coords=[fovs, range(int(rows * scale)), range(int(cols * scale)), channels],
        dims=["fovs", "rows", "cols", "channels"],
    )
    return rescaled_xr


def stitch_images(
    tiff_out_dir,
    run_dir=None,
    channels=None,
    img_sub_folder=None,
    tiled=False,
    intensity_scale=200,
    img_size_scale=0.25,
):
    """Creates a new directory containing stitched channel images for the run
    Args:
        tiff_out_dir (str): path to the extracted images for the specific run
        run_dir (str): path to the run directory containing the run json files, default None
        channels (list): list of channels to produce stitched images for, None will do all
        img_sub_folder (str): optional name of image sub-folder within each fov
        tiled (bool): whether to stitch images back into original tiled shape
        intensity_scale (int): how much to rescale the image values by,
            needed for Photoshop compatibility
        img_size_scale (int/float): amount to scale down image, set to None for no scaling
    """

    io_utils.validate_paths(tiff_out_dir)
    if run_dir:
        io_utils.validate_paths(run_dir)

    # check for previous stitching
    run_name = os.path.basename(tiff_out_dir)
    stitched_dir = os.path.join(tiff_out_dir, f"{run_name}_stitched")
    if tiled:
        stitched_dir = os.path.join(tiff_out_dir, f"{run_name}_tiled")
        if run_dir is None:
            raise ValueError(
                "You must provide the run directory to stitch images into their "
                "original tiled shape."
            )
    if os.path.exists(stitched_dir):
        raise ValueError(f"The stitch_images subdirectory already exists in {tiff_out_dir}")

    folders = io_utils.list_folders(tiff_out_dir, substrs="fov-")
    folders = ns.natsorted(folders)

    # no img_sub_folder, change to empty string to read directly from base folder
    if img_sub_folder is None:
        img_sub_folder = ""

    # retrieve all extracted channel names, or verify the list provided
    if channels is None:
        channels = io_utils.remove_file_extensions(
            io_utils.list_files(
                dir_name=os.path.join(tiff_out_dir, folders[0], img_sub_folder),
                substrs=".tiff",
            )
        )
    else:
        misc_utils.verify_in_list(
            channel_inputs=channels,
            valid_channels=io_utils.remove_file_extensions(
                io_utils.list_files(
                    dir_name=os.path.join(tiff_out_dir, folders[0], img_sub_folder),
                    substrs=".tiff",
                )
            ),
        )

    # get load and stitching args
    tma_folders = None
    if tiled:
        # returns a dict with keys RnCm and values og folder names
        tiled_folders_dict = get_tiled_names(folders, run_dir)
        tma_folders = list(set(folders).difference(set(tiled_folders_dict.values())))
        tma_folders = ns.natsorted(tma_folders)
        try:
            expected_tiles = load_utils.get_tiled_fov_names(
                list(tiled_folders_dict.keys()), return_dims=True
            )
        except AttributeError:
            raise ValueError(f"FOV names found in the run file were not in tiled (RnCm) format.")

    # make stitched subdir
    os.makedirs(stitched_dir)

    # save the stitched images to the stitched_image subdir
    for chan in channels:
        if tiled:
            for tile in expected_tiles:
                prefix, expected_fovs, num_rows, num_cols = tile
                if prefix == "":
                    prefix = "unnamed_tile"
                # subset the folders_dict for fovs found in the current tile
                tile_dict = {
                    fov: tiled_folders_dict[fov]
                    for fov in expected_fovs
                    if fov in tiled_folders_dict.keys()
                }

                # save to individual tile subdirs
                tile_stitched_dir = os.path.join(stitched_dir, prefix)
                if not os.path.exists(tile_stitched_dir):
                    os.makedirs(tile_stitched_dir)

                image_data = load_utils.load_tiled_img_data(
                    tiff_out_dir,
                    tile_dict,
                    expected_fovs,
                    chan,
                    single_dir=False,
                    img_sub_folder=img_sub_folder,
                )
                if img_size_scale:
                    image_data = rescale_stitched_array(image_data, img_size_scale)

                fname = os.path.join(tile_stitched_dir, chan + "_stitched.tiff")
                stitched = data_utils.stitch_images(image_data, num_cols)
                current_img = stitched.loc["stitched_image", :, :, chan] / intensity_scale
                image_utils.save_image(fname, current_img)

        if tma_folders or not tiled:
            # save to individual tma subdir
            if tma_folders:
                folders = tma_folders
                stitched_subdir = os.path.join(stitched_dir, "TMA")
                if not os.path.exists(stitched_subdir):
                    os.makedirs(stitched_subdir)
            else:
                stitched_subdir = stitched_dir

            num_cols = math.isqrt(len(folders))
            max_img_size = get_max_img_size(tiff_out_dir, img_sub_folder, run_dir)

            image_data = load_utils.load_imgs_from_tree(
                tiff_out_dir,
                img_sub_folder=img_sub_folder,
                fovs=folders,
                channels=[chan],
                max_image_size=max_img_size,
            )
            if img_size_scale:
                image_data = rescale_stitched_array(image_data, img_size_scale)

            fname = os.path.join(stitched_subdir, chan + "_stitched.tiff")
            stitched = data_utils.stitch_images(image_data, num_cols)
            current_img = stitched.loc["stitched_image", :, :, chan] / intensity_scale
            image_utils.save_image(fname, current_img)


def rescale_image(img_data, scale, save_path=None):
    """Rescale image data to a desired shape
    Args:
        img_data (np.array): data to be reshaped, expected to only be 2 dimensions
        scale (int): amount to scale the data up or down
        save_path (str): the location to save the tiff file
    Returns:
        numpy.array: data reshaped according to scale value
    """
    # check for 2d data
    if len(img_data.shape) != 2:
        raise ValueError("Image data must only have 2 dimensions.")

    if scale < 1 and img_data.shape[0] % scale != 0:
        raise ValueError("Scale value less than 1 must be a factor of the image size.")

    # rescale data while preserving values
    data_type = img_data.dtype
    rescaled_data = transform.rescale(
        img_data,
        scale,
        mode="constant",
        preserve_range=True,
        order=0,
    )
    rescaled_data = rescaled_data.astype(data_type)

    # overwrite image tiff
    if save_path:
        io_utils.validate_paths(save_path)
        image_utils.save_image(save_path, rescaled_data)

    return rescaled_data


def fix_image_resolutions(resolution_data, extraction_dir):
    """Rescales any images that are a different resolution than the majority in the run
    Args:
        resolution_data (pd.DataFrame): details the fov names and resolutions
        extraction_dir (str): path to the extracted images dir for the specific run
    """
    io_utils.validate_paths(extraction_dir)

    # only check extracted fovs
    extracted_fovs = io_utils.list_folders(extraction_dir, substrs="fov")
    resolution_data = resolution_data[np.isin(resolution_data.fov, extracted_fovs)]

    # identify majority value, subset data for other resolutions
    resolutions = resolution_data["pixels / 400 microns"]
    correct_res = Counter(resolutions).most_common(1)[0][0]
    res_change = resolution_data[resolution_data["pixels / 400 microns"] != correct_res]

    if len(res_change) == 0:
        print("No resolution scaling needed for any FOVs in this run.")

    # loop through problematic fovs
    for fov in res_change.fov:
        fov_res = res_change[res_change.fov == fov]["pixels / 400 microns"].values
        scale = correct_res / fov_res

        tiff_data = load_utils.load_imgs_from_tree(extraction_dir, fovs=[fov])

        # scale and save every channel image
        print(f"Changing {fov} from {int(tiff_data.shape[1])} to {int(tiff_data.shape[1]*scale)}.")
        for channel in tiff_data.channels.values:
            channel_data = np.array(tiff_data.loc[fov, :, :, channel])
            rescale = rescale_image(
                channel_data, scale, save_path=os.path.join(extraction_dir, fov, f"{channel}.tiff")
            )
