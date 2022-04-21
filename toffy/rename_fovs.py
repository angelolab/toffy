import json
import os
import warnings
from distutils.dir_util import copy_tree

from ark.utils import io_utils


def rename_fov_dirs(run_path, fov_dir, new_dir=None):
    """Renames FOV directories with default_name to have custom_name sourced from the run JSON file

    Args:
        run_path (str): path to the JSON run file which contains custom name values
        fov_dir (str): directory where the FOV default named subdirectories are stored
        new_dir (str): path to new directory to output renamed folders and files to, defaults to None

    Raises:
        KeyError: issue reading keys from the JSON file
        ValueError: there are existing default named directories that are not described in the run file
        UserWarning: not all custom names from the run file have an existing directory

        """

    io_utils.validate_paths(run_path)
    io_utils.validate_paths(fov_dir)

    #insert some kind of fov name validation

    #retieve custom names and number of scans for each fov, construct matching default names
    with open(run_path) as f:
        run_metadata = json.load(f)

    fov_scan = dict()
    for fov in run_metadata.get('fovs', ()):
        custom_name = fov.get('name')
        run_order = fov.get('runOrder', -1)
        scans = fov.get('scanCount', -1)
        if run_order * scans < 0:
            raise KeyError(f"Could not locate keys in {run_path}")

        if scans > 1:
            for scan in range(1, scans+1):
                default_name = f'fov-{run_order}-scan-{scan}'
                fov_scan[default_name] = f'{custom_name}-{scan}'
        else:
            default_name = f'fov-{run_order}-scan-{scans}'
            fov_scan[default_name] = custom_name

    #retrieve the current default directory names
    old_dirs = io_utils.list_folders(fov_dir, "fov")

    #check if custom fov names & scan counts match the number of existing default directories
    if not set(old_dirs).issubset(set(fov_scan.keys())):
        raise ValueError(f"FOV folders exceed the expected amount listed in {run_path}")
    if not set(fov_scan.keys()).issubset(set(old_dirs)):
        warnings.warn(f"Not all FOVs listed in {run_path} have an existing directory")

    #validate new_dir and copy contents of fov_dir
    if new_dir is not None:
        io_utils.validate_paths(fov_dir)
        copy_tree(fov_dir, new_dir)
        change_dir = new_dir
    else:
        change_dir = fov_dir

    #change the default directory names to custom names
    for folder in fov_scan:
        fov_subdir = os.path.join(change_dir, folder)
        if os.path.isdir(fov_subdir):
            new_name = os.path.join(change_dir, fov_scan[folder])
            os.rename(fov_subdir, new_name)


'''
r = os.path.join("data", "json_test", "2022-04-07_TONIC_TMA21_run1.json")
#r = os.path.join("data", "json_test", "2022-01-14_postsweep_2.json")
f = os.path.join("data", "fov_folders")
n = os.path.join("data", "new_names")
rename_fov_dirs(r, f, n)'''