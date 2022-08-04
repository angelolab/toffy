import os
import math
import natsort as ns
import skimage.io as io

from toffy import json_utils
from ark.utils import data_utils, load_utils, io_utils, misc_utils
from mibi_bin_tools.io_utils import remove_file_extensions


def get_max_img_size(run_dir):
    """Retrieves the maximum FOV image size listed in the fun file
        Args:
            run_dir (str): path to the run directory containing the run json files """

    run_name = os.path.basename(run_dir)
    run_file_path = os.path.join(run_dir, run_name + '.json')

    # retrieve all pixel width dimensions of the fovs
    run_data = json_utils.read_json_file(run_file_path)
    img_sizes = []
    for fov in run_data['fovs']:
        img_sizes.append(fov.get('frameSizePixels')['width'])

    # return largest
    max_img_size = max(img_sizes)

    return max_img_size


def stitch_images(tiff_out_dir, run_dir, channels=None, img_sub_folder=None):
    """Creates a new directory containing stitched channel images for the run
        Args:
            tiff_out_dir (str): path to the extracted images for the specific run
            run_dir (str): path to the run directory containing the run json files
            channels (list): list of channels to produce stitched images for, None will do all
            img_sub_folder (str): optional name of image sub-folder within each fov"""

    # remove old images
    stitched_dir = os.path.join(tiff_out_dir, 'stitched_images')
    if os.path.exists(stitched_dir):
        raise ValueError(f"fThe stitch_images subdirectory already exists in {tiff_out_dir}")

    folders = io_utils.list_folders(tiff_out_dir)
    folders = ns.natsorted(folders)

    # retrieve all extracted channel names, or verify the list provided
    if channels is None:
        channels = remove_file_extensions(io_utils.list_files(
            dir_name=os.path.join(tiff_out_dir, folders[0]), substrs='.tiff'))
    else:
        misc_utils.verify_in_list(channel_inputs=channels, valid_channels=remove_file_extensions(
            io_utils.list_files(dir_name=os.path.join(tiff_out_dir, folders[0]), substrs='.tiff')))

    # load in and stitch the image data
    num_cols = math.isqrt(len(folders))
    max_img_size = get_max_img_size(run_dir)

    # recreate directory
    os.makedirs(stitched_dir)

    # save the stitched images to the stitched_image subdir
    for chan in channels:
        image_data = load_utils.load_imgs_from_tree(tiff_out_dir, img_sub_folder=img_sub_folder,
                                                    fovs=folders, channels=[chan],
                                                    max_image_size=max_img_size, dtype='float32')
        stitched = data_utils.stitch_images(image_data, num_cols)
        current_img = stitched.loc['stitched_image', :, :, chan].values
        io.imsave(os.path.join(stitched_dir, chan + '_stitched.tiff'),
                  current_img.astype('float32'), check_contrast=False)
