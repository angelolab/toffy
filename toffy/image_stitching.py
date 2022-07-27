import os
import shutil
import math
import natsort as ns
import skimage.io as io

from toffy import json_utils
from ark.utils import data_utils, load_utils, io_utils, misc_utils
from mibi_bin_tools.io_utils import remove_file_extensions


# helper function that scans the run json for max image size, key = frameSize
def get_max_img_size(run_dir):
    run_name = os.path.basename(run_dir)
    run_file_path = os.path.join(run_dir, run_name + '.json')

    run_data = json_utils.read_json_file(run_file_path)
    img_sizes = []
    for fov in run_data['fovs']:
        img_sizes.append(fov.get('frameSizePixels')['width'])

    max_img_size = max(img_sizes)
    return max_img_size


def stitch_images(tiff_out_dir, run_dir, channels=None):

    # remove old images
    stitched_dir = os.path.join(tiff_out_dir, 'stitched_images')
    if os.path.exists(stitched_dir):
        shutil.rmtree(stitched_dir)

    folders = io_utils.list_folders(tiff_out_dir)
    folders = ns.natsorted(folders)

    if channels is None:
        channels = remove_file_extensions(io_utils.list_files(
            dir_name=os.path.join(tiff_out_dir, folders[0]), substrs='.tiff'))
    else:
        misc_utils.verify_in_list(channel_inputs=channels, valid_channels=remove_file_extensions(
            io_utils.list_files(dir_name=os.path.join(tiff_out_dir, folders[0]), substrs='.tiff')))

    num_cols = math.isqrt(len(folders))
    max_img_size = get_max_img_size(run_dir)

    image_data = load_utils.load_imgs_from_tree(tiff_out_dir, fovs=folders, channels=channels,
                                                max_image_size=max_img_size, dtype='uint32')

    stitched = data_utils.stitch_images(image_data, num_cols)

    # recreate directory
    os.makedirs(stitched_dir)

    for chan in stitched.channels.values:
        current_img = stitched.loc['stitched_image', :, :, chan].values
        io.imsave(os.path.join(stitched_dir, chan + '.tiff'), current_img.astype('uint8'),
                  check_contrast=False)
