import copy
import json
import os

from ark.utils import io_utils


def rename_missing_fovs(fov_data):
    """Identify FOVs that are missing the 'name' key and create one with value placeholder_{n}
    Args:
        fov_data (dict): the FOV run JSON data

    Returns:
        dict: a copy of the run JSON data with placeholder names for FOVs that lack one
       """

    copy_fov_data = copy.deepcopy(fov_data)

    # count of FOVs that are missing the 'name' key
    missing_count = 0

    # iterate over each FOV and add a placeholder name if necessary
    for fov in copy_fov_data['fovs']:
        if 'name' not in fov.keys():
            missing_count += 1
            fov['name'] = f'placeholder_{missing_count}'

    return copy_fov_data


def rename_duplicate_fovs(tma_fovs):
    """Identify and rename duplicate FOV names in `fov_list`

    For a given FOV name, the subsequent duplicates get renamed `{FOV}_duplicate{n}`

    Args:
        tma_fovs (dict):
            The TMA run JSON, should contain a `'fovs'` key defining the list of FOVs

    Returns:
        dict:
            The same run JSON with the FOVs renamed to account for duplicates
    """

    # used for identifying the number of times each FOV was found
    fov_count = {}

    # iterate over each FOV
    for fov in tma_fovs['fovs']:
        if fov['name'] not in fov_count:
            fov_count[fov['name']] = 0

        fov_count[fov['name']] += 1

        if fov_count[fov['name']] > 1:
            fov['name'] = '%s_duplicate%d' % (fov['name'], fov_count[fov['name']] - 1)

    return tma_fovs


def list_moly_fovs(bin_file_dir):
    """Lists all of the FOVs in a directory which are moly FOVs

    Args:
        bin_file_dir (str): path to bin files

    Returns:
        list: list of FOVs which are moly FOVs"""

    json_files = io_utils.list_files(bin_file_dir, '.json')
    moly_fovs = []

    for file in json_files:
        json_path = os.path.join(bin_file_dir, file)
        with open(json_path, 'r') as jp:
            json_file = json.load(jp)

        if json_file.get('standardTarget', "") == "Molybdenum Foil":
            moly_name = file.split('.json')[0]
            moly_fovs.append(moly_name)

    return moly_fovs

def read_json_file(json_path):
    """Reads json file and returns json file object while verifying dirs exist

    Args:
        json_path (str): path to json file

    Returns:
        json file object"""

    # call to validate paths will raise errors if anything wrong, and do nothing if 
    # file path valid
    io_utils.validate_paths(json_path,data_prefix=False)

    with open(json_path, 'r') as jp:
        json_file = json.load(jp)

    return json_file

def write_json_file(json_path,json_object):
    """Writes json file object to json file. Raises error if directory doesnt exist.

    Args:
        json_path: full path to write json file
    Returns:
        nothing"""
    
    # get the path minus the proposed file name.
    dir_path = os.path.dirname(os.path.abspath(json_path))

    # Raises error if path doesnt exist
    io_utils.validate_paths(dir_path,data_prefix=False)

    with open(json_path, 'w') as jp:
        json.dump(json_object, jp)

