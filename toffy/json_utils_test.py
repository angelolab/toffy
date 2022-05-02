import numpy as np

from toffy import json_utils, test_utils


def test_rename_missing_fovs():
    # data with missing names
    ex_name = ['custom_1', None, 'custom_2', 'custom_3', None]
    ex_run_order = list(range(1, 6))
    ex_scan_count = list(range(1, 6))

    # create a dict with the sample data
    ex_run = test_utils.create_sample_run(ex_name, ex_run_order, ex_scan_count)

    # test that missing names are given a placeholder
    ex_run = json_utils.rename_missing_fovs(ex_run)
    for fov in ex_run.get('fovs', ()):
        assert fov.get('name') is not None


def test_rename_duplicate_fovs():
    # define a sample set of FOVs
    fov_coords = [(100 + 100 * i, 100 + 100 * j) for i in np.arange(4) for j in np.arange(3)]
    fov_names = ["R%dC%d" % (i, j) for i in np.arange(1, 5) for j in np.arange(1, 4)]
    fov_sizes = [1000] * 12
    fov_list = test_utils.generate_sample_fovs_list(fov_coords, fov_names, fov_sizes)

    # no duplicate FOV names identified, no names should be changed
    fov_list_no_dup = json_utils.rename_duplicate_fovs(fov_list)
    assert [fov['name'] for fov in fov_list_no_dup['fovs']] == fov_names

    # rename R2C2 and R2C3 as R2C1 to create one set of duplicates
    fov_list['fovs'][4]['name'] = 'R2C1'
    fov_list['fovs'][5]['name'] = 'R2C1'
    fov_names[4] = 'R2C1_duplicate1'
    fov_names[5] = 'R2C1_duplicate2'

    fov_list_one_dup = json_utils.rename_duplicate_fovs(fov_list)
    assert [fov['name'] for fov in fov_list_one_dup['fovs']] == fov_names

    # rename R3C3, R4C1, and R4C2 as R3C2 to create another set of duplicates of differing size
    fov_list['fovs'][8]['name'] = 'R3C2'
    fov_list['fovs'][9]['name'] = 'R3C2'
    fov_list['fovs'][10]['name'] = 'R3C2'
    fov_names[8] = 'R3C2_duplicate1'
    fov_names[9] = 'R3C2_duplicate2'
    fov_names[10] = 'R3C2_duplicate3'

    fov_list_mult_dup = json_utils.rename_duplicate_fovs(fov_list)
    assert [fov['name'] for fov in fov_list_mult_dup['fovs']] == fov_names
