import json
import os
import shutil
import warnings
from distutils.dir_util import copy_tree

from ark.utils import io_utils
from ark.utils.misc_utils import verify_in_list
from toffy.json_utils import rename_missing_fovs, rename_duplicate_fovs


def merge_partial_runs(cohort_dir, run_string):
    """Combines different runs together into a single folder of FOVs

    Args:
        cohort_dir (str): the path to the directory containing the run folders
        run_string (str): the substring that each run folder has"""

    # create folder to hold contents of all partial runs
    output_folder = os.path.join(cohort_dir, run_string)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # get list of matching subfolders
    partial_folders = io_utils.list_folders(cohort_dir, substrs=run_string)
    partial_folders = [partial for partial in partial_folders if partial != run_string]

    if len(partial_folders) == 0:
        raise ValueError("No matching folders found for {}".format(run_string))

    # loop through each partial folder
    for partial in partial_folders:
        fov_folders = io_utils.list_folders(os.path.join(cohort_dir, partial))

        # copy each fov from partial folder into output folder
        for fov in fov_folders:
            shutil.copytree(os.path.join(cohort_dir, partial, fov),
                            os.path.join(output_folder, fov))

        # remove partial folder
        shutil.rmtree(os.path.join(cohort_dir, partial))


def combine_runs(cohort_dir):
    """Combines FOVs from different runs together, using the run name as a unique identifier

    Args:
        cohort_dir (str): path to the directory containing individual run folders"""

    # get all runs
    run_folders = io_utils.list_folders(cohort_dir)

    # loop over each run
    for run in run_folders:
        run_path = os.path.join(cohort_dir, run)

        fovs = io_utils.list_folders(run_path)
        for fov in fovs:
            shutil.copytree(os.path.join(run_path, fov),
                            os.path.join(cohort_dir, run + '_' + fov))

        shutil.rmtree(run_path)


def rename_fov_dirs(json_run_path, fov_dir, new_dir=None):
    """Renames FOV directories with default_name to have custom_name sourced from the run JSON file

    Args:
        json_run_path (str): path to the JSON run file which contains the custom name values
        fov_dir (str): directory where the FOV default named subdirectories are stored
        new_dir (str): path to new directory to output renamed folders to, defaults to None

    Raises:
        KeyError: issue reading keys from the JSON file
        ValueError: there exist default named directories that are not described in the run file
        UserWarning: not all custom names from the run file have an existing directory

        """

    io_utils.validate_paths(json_run_path)
    io_utils.validate_paths(fov_dir)

    # check that new_dir doesn't already exist
    if new_dir is not None:
        if os.path.exists(new_dir):
            raise ValueError(f"The new directory supplied already exists: {new_dir}")

    with open(json_run_path) as file:
        run_metadata = json.load(file)

    # check for missing or duplicate fov names
    run_metadata = rename_missing_fovs(run_metadata)
    run_metadata = rename_duplicate_fovs(run_metadata)

    # retrieve custom names and number of scans for each fov, construct matching default names
    fov_scan = dict()
    for fov in run_metadata.get('fovs', ()):
        custom_name = fov.get('name')
        run_order = fov.get('runOrder', -1)
        scans = fov.get('scanCount', -1)
        if run_order * scans < 0:
            raise KeyError(f"Could not locate keys in {run_path}")

        # fovs with multiple scans have scan number specified
        if scans > 1:
            for scan in range(1, scans + 1):
                default_name = f'fov-{run_order}-scan-{scan}'
                fov_scan[default_name] = f'{custom_name}-{scan}'
        else:
            default_name = f'fov-{run_order}-scan-{scans}'
            fov_scan[default_name] = custom_name

    # retrieve current default directory names, check if already renamed
    old_dirs = io_utils.list_folders(fov_dir, 'fov')
    if set(old_dirs) == set(fov_scan.values()):
        raise ValueError(f"All FOV folders in {fov_dir} have already been renamed")

    # check if custom fov names & scan counts match the number of existing default directories
    # verify_in_list(warn=True, fovs_in_run_file=list(fov_scan.keys()),
    #                existing_fov_folders=old_dirs)
    verify_in_list(existing_fov_folders=old_dirs, fovs_in_run_file=list(fov_scan.keys()))

    # validate new_dir and copy contents of fov_dir
    if new_dir is not None:
        copy_tree(fov_dir, new_dir)
        change_dir = new_dir
    else:
        change_dir = fov_dir

    # change the default directory names to custom names
    for folder in fov_scan:
        fov_subdir = os.path.join(change_dir, folder)
        if os.path.isdir(fov_subdir):
            new_name = os.path.join(change_dir, fov_scan[folder])
            os.rename(fov_subdir, new_name)
