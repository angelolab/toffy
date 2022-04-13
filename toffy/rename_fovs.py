import json

from ark.utils import io_utils


def rename_fov_dirs(run_path, fov_dir, new_dir=False):
    """Renames FOV directories to have specific names sources from the run JSON file

    Args:
        run_path (str): path to the JSON run file
        fov_dir (str): directory where the FOV subdirectories are stored
        new_dir (bool): whether or not to output renamed folders to a new directory

    Returns:
        """

    io_utils.validate_paths(run_path)
    io_utils.validate_paths(fov_dir)

    #retieve FOV names and number of scans for each
    with open(run_path) as f:
        data = json.load(f)
        run_data = data['fovs']
        fov_scan = {x['name']: x['scanCount'] for x in run_data}


    #determine how many folders will be changed
    name_count = 0
    for fov in fov_scan:
        name_count += (1*fov_scan[fov])

    #insert some kind of fov name validation


    #retrieve number of current FOV directory and their names
    old_dirs = io_utils.list_folders(fov_dir, "fov")
    dir_count = len(old_dirs)

    #check that fovs & scan counts match the number of existing FOV directories
    if name_count != dir_count:
        raise ValueError(f"FOV folders do not match expected amount listed in the run file")

#rename_fov_dirs("data/json_test/2022-04-07_TONIC_TMA21_run1.json", "data/fov_folders")