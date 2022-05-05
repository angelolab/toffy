import copy


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
