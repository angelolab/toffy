import os
import shutil
import natsort as ns
import skimage.io as io

from ark.utils import data_utils, load_utils, io_utils, misc_utils
from mibi_bin_tools.io_utils import remove_file_extensions


def stitch_images(img_dir, channels=None, max_img_size=2048, num_cols=7):

    # remove old images
    stitched_dir = os.path.join(img_dir, 'stitched_images')
    if os.path.exists(stitched_dir):
        shutil.rmtree(stitched_dir)

    folders = io_utils.list_folders(img_dir)
    folders = ns.natsorted(folders)

    if channels is None:
        channels = remove_file_extensions(io_utils.list_files(
            dir_name=os.path.join(img_dir, folders[0]), substrs='.tiff'))
    else:
        misc_utils.verify_in_list(channel_inputs=channels, valid_channels=remove_file_extensions(
            io_utils.list_files(dir_name=os.path.join(img_dir, folders[0]), substrs='.tiff')))

    qc_fovs = []
    for folder in folders:
        img = io.imread(os.path.join(img_dir, folder, channels[0]+'.tiff'))
        if img.shape[0] == 128:
            qc_fovs.append(folder)
    folders = [folder for folder in folders if folder not in qc_fovs]

    image_data = load_utils.load_imgs_from_tree(img_dir, fovs=folders, channels=channels,
                                                max_image_size=max_img_size)
    stitched = data_utils.stitch_images(image_data, num_cols)

    # recreate directory
    os.makedirs(stitched_dir)

    for chan in stitched.channels.values:
        current_img = stitched.loc['stitched_image', :, :, chan].values
        io.imsave(os.path.join(stitched_dir, chan + '.tiff'), current_img.astype('uint8'),
                  check_contrast=False)
