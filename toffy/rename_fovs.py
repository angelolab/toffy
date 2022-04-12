import json

from ark.utils import io_utils


def rename_fov_dirs(run_dir, fov_dir, new_dir=FALSE):
    """Renames FOV directories to have specific names sources from the run JSON file

    Args:
        run_dir (str): directory with the json run file
        fov_dir (str): directory where the FOV subdirectories are stored
        new_dir (bool): whether or not to output renamed folders to a new directory

    Returns:
        """

