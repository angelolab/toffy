import copy
import json
import os
import warnings

from ark.utils import io_utils
from mibi_bin_tools.io_utils import remove_file_extensions


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


def list_moly_fovs(bin_file_dir, fov_list=None):
    """Lists all of the FOVs in a directory which are moly FOVs

    Args:
        bin_file_dir (str): path to bin files
        fov_list (list): list of fov names to check

    Returns:
        list: list of FOVs which are moly FOVs"""

    # check provided fovs
    if fov_list:
        json_files = [fov + '.json' for fov in fov_list]
    # check all fovs iin bin_file_dir
    else:
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


def read_json_file(json_path, encoding=None):
    """Reads json file and returns json file object while verifying dirs exist
    Args:
        json_path (str): path to json file
    Returns:
        json file object"""

    # call to validate paths will raise errors if anything wrong, and do nothing if
    # file path valid
    io_utils.validate_paths(json_path, data_prefix=False)

    with open(json_path, mode='r', encoding=encoding) as jp:
        json_file = json.load(jp)

    return json_file


def write_json_file(json_path, json_object, encoding=None):
    """Writes json file object to json file. Raises error if directory doesnt exist.
    Args:
        json_path: full path to write json file
    Returns:
        nothing"""

    # get the path minus the proposed file name.
    dir_path = os.path.dirname(os.path.abspath(json_path))

    # Raises error if path doesnt exist
    io_utils.validate_paths(dir_path, data_prefix=False)

    with open(json_path, mode='w', encoding=encoding) as jp:
        json.dump(json_object, jp)


def split_run_file(run_dir, run_file_name, file_split: list):
    """Splits a run json file into smaller fov amount files as defined by the user

    Args:
        run_dir (str): path to directory containing the run file
        run_file_name (str): name of the run file to split
        file_split (list): list of ints defining how to break up the fovs into new jsons

    Returns:
        saves the new json files to the base_dir """

    json_path = os.path.join(run_dir, run_file_name)
    full_json = read_json_file(json_path, encoding='utf-8')

    # check list is valid FOV split
    if not sum(file_split) == len(full_json['fovs']):
        raise ValueError(
            f"Sum of the provided list does not match the number of FOVs in the run file.\n"
            f"list sum: {sum(file_split)}, total FOVs in the JSON: {len(full_json['fovs'])}")

    # split the run json into smaller files and save to run_dir
    start = 0
    for i in range(0, len(file_split)):
        json_i = copy.deepcopy(full_json)
        stop = start+file_split[i]
        json_i['fovs'] = json_i['fovs'][start:stop]
        start = start+file_split[i]

        save_path = os.path.join(run_dir, run_file_name.split('.json')[0] + '_part'
                                 + str(i+1) + '.json')
        write_json_file(save_path, json_i, encoding='utf-8')


def check_for_empty_files(bin_file_dir, return_json_names=False, warn=True):
    """ Check for any empty json files and warn the user
    Args:
        bin_file_dir (str): directory containing the bin and json files
        return_json_names (bool): whether to return a list of fovs with empty json files
        warn (bool): whether to print a warning to the user

    Return:
        (list) of fov files with empty json, if none returns empty list
        raises a warning
    """

    # retrieve all fovs in bin_file_dir
    fov_names = remove_file_extensions(io_utils.list_files(bin_file_dir, substrs='.bin'))
    empty_json_files = []

    # check each json file for size 0
    for fov in fov_names:
        fov_path = os.path.join(bin_file_dir, fov + '.json')
        if os.path.getsize(fov_path) == 0:
            empty_json_files.append(fov)

    # print a warning to the user when there are empty files
    if warn and empty_json_files:
        warnings.warn(f'The following FOVs have empty json files and will not be processed:'
                      f'\n {empty_json_files}', UserWarning)

    # return the list of fov names
    if return_json_names:
        return empty_json_files
