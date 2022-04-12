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

    io_utils.validate_paths(run_dir)
    io_utils.validate_paths(run_dir)

    #retieve FOV names and number of scans for each
    with open(run_dir) as f:
        data = json.load(f)
        run_data = data['fovs']
        fov_scan = {x['name']: x['scanCount'] for x in run_data}