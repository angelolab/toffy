import os
import pathlib
import shutil
from stat import S_IREAD, S_IRGRP, S_IROTH

import pytest

from toffy.utils import remove_readonly


@pytest.fixture(scope="function")
def create_readonly_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Creates a readonly file.

    Args:
        tmp_path (pathlib.Path): A temporary path to create the file in.

    Returns:
        pathlib.Path: The path to the file.
    """

    ro_path: pathlib.Path = tmp_path / "ro_dir"
    ro_path.mkdir()

    ro_file: pathlib.Path = ro_path / "ro_file.txt"
    ro_file.write_text("This is a readonly file.")
    os.chmod(ro_file, S_IREAD | S_IRGRP | S_IROTH)
    yield ro_path


def test_remove_readonly(create_readonly_file: pathlib.Path):
    assert not os.access(create_readonly_file / "ro_file.txt", os.W_OK)

    shutil.rmtree(create_readonly_file, onerror=remove_readonly)
    assert not (create_readonly_file / "ro_file.txt").exists()
