import os
import shutil
import tempfile

import numpy as np
import pytest
from alpineer.image_utils import save_image

from toffy import file_hash


def test_get_hash():
    with tempfile.TemporaryDirectory() as temp_dir:
        for img in range(2):
            array = np.random.rand(36).reshape((6, 6))
            temp_file_path = os.path.join(temp_dir, "test_file_{}.tiff".format(img))
            save_image(temp_file_path, array)

        shutil.copy(
            os.path.join(temp_dir, "test_file_0.tiff"),
            os.path.join(temp_dir, "test_file_0_copy.tiff"),
        )

        hash1 = file_hash.get_hash(os.path.join(temp_dir, "test_file_0.tiff"))
        hash1_copy = file_hash.get_hash(os.path.join(temp_dir, "test_file_0_copy.tiff"))
        hash2 = file_hash.get_hash(os.path.join(temp_dir, "test_file_1.tiff"))

        assert hash1 != hash2
        assert hash1 == hash1_copy


def test_compare_directories():
    with tempfile.TemporaryDirectory() as top_level_dir:
        dir_1 = os.path.join(top_level_dir, "dir_1")
        os.makedirs(dir_1)

        # make fake data for testing
        for img in range(5):
            array = np.random.rand(36).reshape((6, 6))
            temp_file_path = os.path.join(dir_1, "test_file_{}.tiff".format(img))
            save_image(temp_file_path, array)

        # copy same data into second directory
        dir_2 = os.path.join(top_level_dir, "dir_2")
        shutil.copytree(dir_1, dir_2)

        file_hash.compare_directories(dir_1, dir_2)

        # check that warning is raised when sub-folder is present in first directory
        sub_folder_1 = os.path.join(dir_1, "sub_folder")
        os.makedirs(sub_folder_1)

        with pytest.warns(UserWarning, match="first directory"):
            file_hash.compare_directories(dir_1, dir_2)

        # check that warning is raised when sub-folder is present in second directory
        shutil.rmtree(sub_folder_1)
        sub_folder_2 = os.path.join(dir_2, "sub_folder")
        os.makedirs(sub_folder_2)

        with pytest.warns(UserWarning, match="second directory"):
            file_hash.compare_directories(dir_1, dir_2)
