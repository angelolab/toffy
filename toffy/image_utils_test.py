import os
import pathlib
from typing import Iterator

import numpy as np
import pytest
import skimage.io as io

from toffy import image_utils


@pytest.fixture(scope="session")
def create_img_data() -> Iterator[np.ndarray]:
    """
    A Fixture which creates a numpy array for tiff file compression testing.

    Returns:
        Iterator[np.ndarray]: Returns a randomly generated (1000 x 1000) numpy array.
    """

    # Initialize a new generator - set seed for reproducibility
    rng = np.random.default_rng(12345)

    # Create testing data array
    data: np.ndarray = rng.integers(low=0, high=256, size=(1000, 1000), dtype=np.int16)

    yield data


class TestSaveImage:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, create_img_data):
        self.uncompressed_fname: pathlib.Path = tmp_path / "test_img.tiff"
        self.compressed_fname: pathlib.Path = tmp_path / "test_img_compressed.tiff"
        self.data: np.ndarray = create_img_data

        # save uncompressed image
        io.imsave(
            self.uncompressed_fname,
            arr=self.data,
            plugin="tifffile",
            check_contrast=False,
        )

    @pytest.mark.parametrize("compress_level", [1, 6, 9,
                                                pytest.param(10, marks=pytest.mark.xfail)])
    def test_save_compressed_img(self, compress_level):
        # Fails when compression_level > 9
        image_utils.save_image(
            fname=self.compressed_fname, data=self.data, compression_level=compress_level
        )

        # Assert that the compressed tiff file is smaller than the uncompressed tiff file
        uncompressed_tiff_file_size: int = os.path.getsize(self.uncompressed_fname)
        compressed_tiff_file_size: int = os.path.getsize(self.compressed_fname)

        assert compressed_tiff_file_size < uncompressed_tiff_file_size

        # Assert that the values in the compressed tiff file and the uncompressed
        # tiff file are equal.

        uncompressed_data: np.ndarray = io.imread(self.uncompressed_fname)
        compressed_data: np.ndarray = io.imread(self.compressed_fname)

        np.testing.assert_array_equal(compressed_data, uncompressed_data)
