import os
import tempfile
import pytest
import shutil

from toffy import image_stitching, json_utils
from ark.utils import io_utils, test_utils


def test_get_max_img_size():

    RUN_JSON_SPOOF = {
        'fovs': [
            {'runOrder': 1, 'scanCount': 1, 'frameSizePixels': {'width': 32, 'height': 32}},
            {'runOrder': 2, 'scanCount': 1, 'frameSizePixels': {'width': 16, 'height': 16}},
            {'runOrder': 3, 'scanCount': 1, 'frameSizePixels': {'width': 8, 'height': 8}},
            {'runOrder': 4, 'scanCount': 1, 'frameSizePixels': {'width': 16, 'height': 16}},
        ],
    }

    # test with run file
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = os.path.join(tmp_dir, 'data', 'test_run')
        os.makedirs(test_dir)
        json_path = os.path.join(test_dir, 'test_run.json')
        json_utils.write_json_file(json_path, RUN_JSON_SPOOF)

        # test success for all fovs
        max_img_size = image_stitching.get_max_img_size('extracted_dir', test_dir)
        assert max_img_size == 32

        # test success for fov list
        max_img_size = image_stitching.get_max_img_size('extracted_dir', test_dir,
                                                        ['fov-2-scan-1', 'fov-3-scan-1'])
        assert max_img_size == 16

    # test by reading image sizes
    with tempfile.TemporaryDirectory() as tmpdir:
        channel_list = ['Au', 'CD3', 'CD4', 'CD8', 'CD11c']
        fov_list = ['fov-1-scan-1', 'fov-2-scan-1']
        larger_fov = ['fov-3-scan-1']

        test_utils._write_tifs(tmpdir, fov_list, channel_list, (16, 16), '', False, int)
        test_utils._write_tifs(tmpdir, larger_fov, channel_list, (32, 32), '', False, int)

        # test success for all fovs
        max_img_size = image_stitching.get_max_img_size(tmpdir)
        assert max_img_size == 32

        # test success for fov list
        max_img_size = image_stitching.get_max_img_size(tmpdir, 'bin_dir',
                                                        ['fov-1-scan-1', 'fov-2-scan-1'])
        assert max_img_size == 16


def test_stitch_images(mocker):
    mocker.patch('toffy.image_stitching.get_max_img_size', return_value=32)

    channel_list = ['Au', 'CD3', 'CD4', 'CD8', 'CD11c']
    stitched_tifs = ['Au_stitched.tiff', 'CD3_stitched.tiff', 'CD4_stitched.tiff',
                     'CD8_stitched.tiff', 'CD11c_stitched.tiff']
    fov_list = ['fov-1-scan-1', 'fov-2-scan-1', 'fov-3-scan-1']

    with tempfile.TemporaryDirectory() as tmpdir:
        test_utils._write_tifs(tmpdir, fov_list, channel_list, (10, 10), '', False, int)

        # bad channel should raise an error
        with pytest.raises(ValueError, match='Not all values given in list channel inputs were '
                                             'found in list valid channels.'):
            image_stitching.stitch_images(tmpdir, tmpdir, ['Au', 'bad_channel'])

        # test successful stitching for all channels
        image_stitching.stitch_images(tmpdir, tmpdir)
        assert sorted(io_utils.list_files(os.path.join(tmpdir, 'stitched_images'))) == \
               sorted(stitched_tifs)

        # test previous stitching raises an error
        with pytest.raises(ValueError, match="The stitch_images subdirectory already exists"):
            image_stitching.stitch_images(tmpdir, tmpdir)
        shutil.rmtree(os.path.join(tmpdir, 'stitched_images'))

        # test stitching for specific channels
        image_stitching.stitch_images(tmpdir, channels=['Au', 'CD3'])
        assert sorted(io_utils.list_files(os.path.join(tmpdir, 'stitched_images'))) == \
               sorted(['Au_stitched.tiff', 'CD3_stitched.tiff'])
        shutil.rmtree(os.path.join(tmpdir, 'stitched_images'))

    with tempfile.TemporaryDirectory() as tmpdir:
        test_utils._write_tifs(tmpdir, fov_list, channel_list, (10, 10), 'sub_dir', False, int)

        # test stitching for images in subdir
        image_stitching.stitch_images(tmpdir, tmpdir, ['Au', 'CD3'], img_sub_folder='sub_dir')
        assert sorted(io_utils.list_files(os.path.join(tmpdir, 'stitched_images'))) == \
               sorted(['Au_stitched.tiff', 'CD3_stitched.tiff'])
        shutil.rmtree(os.path.join(tmpdir, 'stitched_images'))

        # test stitching for no run_dir
        image_stitching.stitch_images(tmpdir, channels=['Au', 'CD3'], img_sub_folder='sub_dir')
        assert sorted(io_utils.list_files(os.path.join(tmpdir, 'stitched_images'))) == \
               sorted(['Au_stitched.tiff', 'CD3_stitched.tiff'])
