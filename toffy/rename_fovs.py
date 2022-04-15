import json
import os
from distutils.dir_util import copy_tree

from ark.utils import io_utils


def rename_fov_dirs(run_path, fov_dir, new_dir=None):
    """Renames FOV directories to have specific names sources from the run JSON file

    Args:
        run_path (str): path to the JSON run file
        fov_dir (str): directory where the FOV subdirectories are stored
        new_dir (str): name of new directory to output files to, defaults to None

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


    #retrieve number of current FOV directories and their names
    old_dirs = io_utils.list_folders(fov_dir, "fov")
    dir_count = len(old_dirs)
    old_dirs.sort()

    #testing
    #test = list(fov_scan)[0:5]
    #fov_scan = {t: fov_scan[t] for t in test}
    #print(fov_scan)

    #check that fovs & scan counts match the number of existing FOV directories
    if name_count != dir_count:
        raise ValueError(f"FOV folders do not match expected amount listed in the run file")

    # create new directory and copy contents of fov_dir
    if new_dir is not None:
        parent = os.path.join(fov_dir, os.pardir)
        new_dir = os.path.join(parent, new_dir)
        copy_tree(fov_dir, new_dir)

    #change the FOV directory names
    renamed_dirs = 0
    for fov in fov_scan:
        scan_num = fov_scan[fov]
        for scan in range(1, scan_num+1):
            if new_dir is not None:
                change_dir = new_dir
            else:
                change_dir = fov_dir
            fov_subdir = os.path.join(change_dir, old_dirs[renamed_dirs])
            new_name = os.path.join(change_dir, fov + "-scan-" + str(scan))
            os.rename(fov_subdir, new_name)
            renamed_dirs += 1


'''
r = os.path.join("data", "json_test", "2022-04-07_TONIC_TMA21_run1.json")
f = os.path.join("data", "fov_folders")
n = 'new_directory_name'
rename_fov_dirs(r, f, n)'''