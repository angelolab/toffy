import os
import tempfile
import pytest

from toffy import image_stitching
from ark.utils import io_utils, test_utils


def test_stitch_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        channel_list = ['Au', 'CD3', 'CD4', 'CD8', 'CD11c']
        stitched_tifs = ['Au.tiff', 'CD3.tiff', 'CD4.tiff', 'CD8.tiff', 'CD11c.tiff']
        fov_list = ['fov-1-scan-1', 'fov-2-scan-1', 'fov-3-scan-1']
        test_utils._write_tifs(tmpdir, fov_list, channel_list, (10, 10), '', False, int)

        # bad channel should raise an error
        with pytest.raises(ValueError, match='Not all values given in list channel inputs were '
                                             'found in list valid channels.'):
            image_stitching.stitch_images(tmpdir, ['Au', 'bad_channel'])

        # test successful stitching for all channels
        image_stitching.stitch_images(tmpdir)
        assert sorted(io_utils.list_files(os.path.join(tmpdir, 'stitched_images'))) == \
               sorted(stitched_tifs)

        # test stitching for specific channels
        image_stitching.stitch_images(tmpdir, ['Au', 'CD3'])
        assert sorted(io_utils.list_files(os.path.join(tmpdir, 'stitched_images'))) == \
               sorted(['Au.tiff', 'CD3.tiff'])