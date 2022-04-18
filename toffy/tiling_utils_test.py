import copy
from inspect import cleandoc
import ipywidgets as widgets
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytest
from pytest_cases import parametrize_with_cases
import random
from sklearn.utils import shuffle
import tempfile

from dataclasses import dataclass, astuple

from toffy import settings
from toffy import test_utils
from toffy import tiling_utils
from toffy import tiling_utils_test_cases as test_cases

from ark.utils import misc_utils

param = pytest.param
parametrize = pytest.mark.parametrize
xfail = pytest.mark.xfail

# shortcuts to make the marks arg in pytest.params easier
file_missing_err = [xfail(raises=FileNotFoundError, strict=True)]
value_err = [xfail(raises=ValueError, strict=True)]


def test_assign_metadata_vals():
    example_input_dict = {
        1: 'hello',
        2: False,
        3: 5.1,
        4: 7,
        5: {'do': 'not copy'},
        6: ['blah'],
        7: None
    }

    example_output_dict = {
        3: True,
        4: 1,
        5: {'hello': 'world'},
        6: 3.14,
        7: None
    }

    example_keys_ignore = [2, 4, 6, 8]

    new_output_dict = tiling_utils.assign_metadata_vals(
        example_input_dict, example_output_dict, example_keys_ignore
    )

    # assert the keys are correct
    misc_utils.verify_same_elements(
        new_output_keys=list(new_output_dict.keys()),
        valid_keys=[1, 3, 4, 5, 6, 7]
    )

    # assert the values in each key is correct
    assert new_output_dict[3] == 5.1
    assert new_output_dict[4] == 1
    assert 'hello' in new_output_dict[5] and new_output_dict[5]['hello'] == 'world'
    assert new_output_dict[6] == 3.14
    assert new_output_dict[7] is None


def test_read_tiling_param(monkeypatch):
    # test an invalid non-int response, an invalid int response, then a valid response
    user_inputs_int = iter(['N', 0, 1])

    # make sure the function receives the incorrect input first then the correct input
    monkeypatch.setattr('builtins.input', lambda _: next(user_inputs_int))

    # simulate the input sequence for int
    sample_tiling_param = tiling_utils.read_tiling_param(
        "Sample prompt: ",
        "Sample error message",
        lambda x: x == 1,
        dtype=int
    )

    # assert sample_tiling_param was set to 1
    assert sample_tiling_param == 1

    # test an invalid non-str response, then an invalid str response, then a valid response
    user_inputs_str = iter([1, 'N', 'Y'])

    # make sure the function receives the incorrect input first then the correct input
    monkeypatch.setattr('builtins.input', lambda _: next(user_inputs_str))

    # simulate the input sequence for str
    sample_tiling_param = tiling_utils.read_tiling_param(
        "Sample prompt: ",
        "Sample error message",
        lambda x: x == 'Y',
        dtype=str
    )

    # assert sample_tiling_param was set to 'Y'
    assert sample_tiling_param == 'Y'


@parametrize_with_cases('user_inputs', cases=test_cases.FiducialInfoReadCases)
def test_read_fiducial_info(monkeypatch, user_inputs):
    # generate the user inputs
    user_inputs = iter(user_inputs)

    # override the default functionality of the input function
    monkeypatch.setattr('builtins.input', lambda _: next(user_inputs))

    # use the dummy user data to read the fiducial values
    fiducial_info = tiling_utils.read_fiducial_info()

    # verify that the stage coordinates are correct
    fiducial_stage_x = [fiducial_info['stage'][pos]['x'] for pos in settings.FIDUCIAL_POSITIONS]
    assert fiducial_stage_x == [1.5 + 6 * i for i in np.arange(6)]

    fiducial_stage_y = [fiducial_info['stage'][pos]['y'] for pos in settings.FIDUCIAL_POSITIONS]
    assert fiducial_stage_y == [2 + 8 * i for i in np.arange(6)]

    # verify that the optical coordinates are correct
    fiducial_pixel_x = [fiducial_info['optical'][pos]['x'] for pos in settings.FIDUCIAL_POSITIONS]
    assert fiducial_pixel_x == [4.5 + 6 * i for i in np.arange(6)]

    fiducial_pixel_y = [fiducial_info['optical'][pos]['y'] for pos in settings.FIDUCIAL_POSITIONS]
    assert fiducial_pixel_y == [6 + 8 * i for i in np.arange(6)]


def test_generate_coreg_params():
    # define a sample fiducial info dict
    sample_fiducial_info = {
        'stage': {
            pos: {'x': i * 2 + 1, 'y': i * 3 + 1}
            for (i, pos) in enumerate(settings.FIDUCIAL_POSITIONS)
        },
        'optical': {
            pos: {'x': i * 4 + 1, 'y': i * 9 + 1}
            for (i, pos) in enumerate(settings.FIDUCIAL_POSITIONS)
        }
    }

    # generate the regression parameters
    sample_coreg_params = tiling_utils.generate_coreg_params(sample_fiducial_info)

    # assert the computed regression parameters are correct
    # NOTE: values need floating point correction due to how it's calculated
    assert round(sample_coreg_params['STAGE_TO_OPTICAL_X_MULTIPLIER'], 1) == 2
    assert round(sample_coreg_params['STAGE_TO_OPTICAL_X_OFFSET'], 1) == -0.5
    assert round(sample_coreg_params['STAGE_TO_OPTICAL_Y_MULTIPLIER'], 1) == 3
    assert round(sample_coreg_params['STAGE_TO_OPTICAL_Y_OFFSET'], 1) == -0.7


def test_save_coreg_params():
    with tempfile.TemporaryDirectory() as temp_dir:
        # make a dummy toffy directory and a dummy templates directory where the code is run
        os.mkdir(os.path.join(temp_dir, 'toffy'))
        os.mkdir(os.path.join(temp_dir, 'templates'))

        # change working directory to templates to simulate actual co-registration run
        os.chdir(os.path.join(temp_dir, 'templates'))

        # test saving the first time
        sample_coreg_params_first = {
            'STAGE_TO_OPTICAL_X_MULTIPLIER': 2,
            'STAGE_TO_OPTICAL_X_OFFSET': -0.5,
            'STAGE_TO_OPTICAL_Y_MULTIPLIER': 3,
            'STAGE_TO_OPTICAL_Y_OFFSET': -0.7,
            'date': '22/03/2022 00:00:00'
        }
        tiling_utils.save_coreg_params(sample_coreg_params_first)

        # assert we actually created coreg_params.json in toffy
        assert os.path.exists(os.path.join('..', 'toffy', 'coreg_params.json'))

        # load the first co-registration save data in
        with open(os.path.join('..', 'toffy', 'coreg_params.json'), 'r') as cp:
            coreg_data = json.load(cp)

        # assert 1 element in the coreg_params key and it contains the right coreg vals
        assert len(coreg_data['coreg_params']) == 1
        assert coreg_data['coreg_params'][0] == sample_coreg_params_first

        # test saving the second time (should only append)
        sample_coreg_params_second = {
            'STAGE_TO_OPTICAL_X_MULTIPLIER': 4,
            'STAGE_TO_OPTICAL_X_OFFSET': -1,
            'STAGE_TO_OPTICAL_Y_MULTIPLIER': 6,
            'STAGE_TO_OPTICAL_Y_OFFSET': -1.4,
            'date': '23/03/2022 00:00:00'
        }
        tiling_utils.save_coreg_params(sample_coreg_params_second)

        # load the second co-registration save data in
        # NOTE: since the previous step only appended, coreg_params.json will not disappear
        with open(os.path.join('..', 'toffy', 'coreg_params.json'), 'r') as cp:
            coreg_data = json.load(cp)

        # assert 2 elements in the coreg_params key and they contain the right coreg vals
        assert len(coreg_data['coreg_params']) == 2
        assert coreg_data['coreg_params'][0] == sample_coreg_params_first
        assert coreg_data['coreg_params'][1] == sample_coreg_params_second


@parametrize_with_cases(
    'fov_coords, fov_names, fov_sizes, user_inputs, base_param_values, full_param_set',
    cases=test_cases.TiledRegionReadCases, glob='*_no_moly_param')
def test_read_tiled_region_inputs(monkeypatch, fov_coords, fov_names, fov_sizes, user_inputs,
                                  base_param_values, full_param_set):
    # define a sample fovs list to define the top-left corners of each tiled region
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=fov_coords, fov_names=fov_names, fov_sizes=fov_sizes
    )

    # define sample_region_params to read data into
    sample_region_params = {rpf: [] for rpf in settings.REGION_PARAM_FIELDS}

    # generate the user inputs
    user_inputs = iter(user_inputs)

    # override the default functionality of the input function
    monkeypatch.setattr('builtins.input', lambda _: next(user_inputs))

    # use the dummy user data to read values into the params lists
    tiling_utils.read_tiled_region_inputs(
        sample_fovs_list, sample_region_params
    )

    # assert all the keys are added (region_start_x, region_start_y, num_x, num_y, etc.)
    misc_utils.verify_same_elements(
        tiling_keys=list(sample_region_params.keys()),
        required_keys=list(full_param_set.keys())
    )

    # test the value of each tiling parameter
    # NOTE: since both sample_region_params and param_values are in param: [list] format
    # just test equality
    for param in sample_region_params:
        assert sample_region_params[param] == full_param_set[param]


# NOTE: since this is just a "copy over" function, don't need to parametrize
def test_generate_region_info():
    # retrieve the base params and the full set of params (use the defaults for consistency)
    base_param_values, full_param_set = test_cases.generate_tiled_region_params()

    # add region_rand to full_param set
    full_param_set['region_rand'] = ['N', 'Y']

    # generate the region params (full_param_set is in the sample input format so use that)
    sample_region_info = tiling_utils.generate_region_info(full_param_set)

    # test the individual values, using the start values of each numeric param as a base
    # NOTE: we test region_rand separately since it's a string
    for i in range(len(sample_region_info)):
        for param, val in list(sample_region_info[i].items()):
            if not isinstance(val, str):
                assert val == base_param_values[param] * (i + 1)

    # assert region_rand set correctly
    assert sample_region_info[0]['region_rand'] == 'N'
    assert sample_region_info[1]['region_rand'] == 'Y'


# NOTE: you can use this to assert failures without needing a separate test class
@parametrize('region_corners_file', [param('bad_region_corners.json', marks=file_missing_err),
                                     param('tiled_region_corners.json')])
@parametrize_with_cases(
    'fov_coords, fov_names, fov_sizes, user_inputs, base_param_values, full_param_set',
    cases=test_cases.TiledRegionReadCases, glob='*_with_moly_param'
)
@parametrize('moly_interval_val', [0, 1])
def test_set_tiled_region_params(monkeypatch, region_corners_file, fov_coords, fov_names,
                                 fov_sizes, user_inputs, base_param_values,
                                 full_param_set, moly_interval_val):
    # define a sample set of fovs to define the top-left corners of each tiled region
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=fov_coords, fov_names=fov_names, fov_sizes=fov_sizes
    )

    # set the user inputs
    user_inputs = iter(user_inputs + [moly_interval_val])

    # override the default functionality of the input function
    monkeypatch.setattr('builtins.input', lambda _: next(user_inputs))

    with tempfile.TemporaryDirectory() as temp_dir:
        # write fov list
        sample_fov_list_path = os.path.join(temp_dir, 'tiled_region_corners.json')
        with open(sample_fov_list_path, 'w') as fl:
            json.dump(sample_fovs_list, fl)

        # run tiling parameter setting process with predefined user inputs
        sample_tiling_params = tiling_utils.set_tiled_region_params(
            os.path.join(temp_dir, region_corners_file)
        )

        # assert the fovs in the tiling params are the same as in the original fovs list
        assert sample_tiling_params['fovs'] == sample_fovs_list['fovs']

        # assert region start x and region start y values are correct
        sample_region_info = sample_tiling_params['region_params']

        # test the value of each numeric tiling param for both regions
        for i in range(len(sample_region_info)):
            for param, val in list(sample_region_info[i].items()):
                if not isinstance(val, str):
                    assert val == base_param_values[param] * (i + 1)

        # assert region_rand set correctly
        assert sample_region_info[0]['region_rand'] == 'N'
        assert sample_region_info[1]['region_rand'] == 'Y'

        # assert moly_region set properly
        assert sample_tiling_params['moly_region'] == 'Y'

        # if moly_interval set to 0 assert it doesn't exist,
        # otherwise ensure it exists and it's set to 1
        if moly_interval_val == 0:
            assert 'moly_interval' not in sample_tiling_params
        else:
            assert 'moly_interval' in sample_tiling_params
            assert sample_tiling_params['moly_interval'] == 1


def test_generate_x_y_fov_pairs():
    # define sample x and y pair lists
    sample_x_range = [0, 5]
    sample_y_range = [2, 4]

    # generate the sample (x, y) pairs
    sample_pairs = tiling_utils.generate_x_y_fov_pairs(sample_x_range, sample_y_range)

    assert sample_pairs == [(0, 2), (0, 4), (5, 2), (5, 4)]


@parametrize_with_cases('coords, actual_pairs', cases=test_cases.RhombusCoordInputCases)
def test_generate_x_y_fov_pairs_rhombus(coords, actual_pairs):
    # retrieve the coordinates defining the TMA and the number of FOVs along each axis
    top_left, top_right, bottom_left, bottom_right = coords

    # generate the FOV-coordinate pairs
    pairs = tiling_utils.generate_x_y_fov_pairs_rhombus(
        top_left, top_right, bottom_left, bottom_right, 3, 4
    )

    assert pairs == actual_pairs


@parametrize_with_cases(
    'moly_path, moly_roi_setting, moly_interval_setting, moly_interval_value, '
    'moly_insert_indices, roi_1_end_pos', cases=test_cases.TiledRegionMolySettingCases
)
@parametrize('randomize_setting', [['N', 'N'], ['N', 'Y'], ['Y', 'Y']])
def test_generate_tiled_region_fov_list(moly_path, moly_roi_setting,
                                        moly_interval_setting, moly_interval_value,
                                        moly_insert_indices, roi_1_end_pos, randomize_setting):
    # define a set of fovs defining the upper-left corners of each region
    sample_roi_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=[(0, 0), (100, 100)], fov_names=['TheFirstROI', 'TheSecondROI'],
        fov_sizes=[5, 10]
    )

    sample_region_inputs = {
        'region_name': ['TheFirstROI', 'TheSecondROI'],
        'region_start_row': [100, 150],
        'region_start_col': [0, 50],
        'fov_num_row': [2, 4],
        'fov_num_col': [4, 2],
        'row_fov_size': [5, 10],
        'col_fov_size': [5, 10],
        'region_rand': ['N', 'N']
    }

    sample_region_params = tiling_utils.generate_region_info(sample_region_inputs)

    sample_tiling_params = {
        'fovFormatVersion': '1.5',
        'fovs': sample_roi_fovs_list['fovs'],
        'region_params': sample_region_params
    }

    with tempfile.TemporaryDirectory() as td:
        sample_moly_point = test_utils.generate_sample_fov_tiling_entry(
            coord=(14540, -10830), name="MoQC", size=10000
        )
        sample_moly_path = os.path.join(td, 'sample_moly_point.json')

        with open(sample_moly_path, 'w') as smp:
            json.dump(sample_moly_point, smp)

        sample_tiling_params['moly_region'] = moly_roi_setting

        sample_tiling_params['region_params'][0]['region_rand'] = randomize_setting[0]
        sample_tiling_params['region_params'][1]['region_rand'] = randomize_setting[1]

        if moly_interval_setting:
            sample_tiling_params['moly_interval'] = moly_interval_value

        fov_list = tiling_utils.generate_tiled_region_fov_list(
            sample_tiling_params, os.path.join(td, moly_path)
        )

        # assert none of the metadata keys explicitly added by set_tiling_params appear
        for k in ['region_params', 'moly_region', 'moly_interval']:
            assert k not in fov_list

        # retrieve the center points
        center_points = [
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            for fov in fov_list['fovs']
        ]

        # retrieve the fov names
        fov_names = [
            fov['name'] for fov in fov_list['fovs']
        ]

        # define the center points sorted
        actual_center_points_sorted = [
            (x, y) for x in np.arange(0, 10, 5) for y in list(reversed(np.arange(85, 105, 5)))
        ] + [
            (x, y) for x in np.arange(50, 90, 10) for y in list(reversed(np.arange(140, 160, 10)))
        ]

        # define the corresponding FOV names
        actual_fov_names = [
            'TheFirstROI_R%dC%d' % (x, y) for x in np.arange(1, 3) for y in np.arange(1, 5)
        ] + [
            'TheSecondROI_R%dC%d' % (x, y) for x in np.arange(1, 5) for y in np.arange(1, 3)
        ]

        for mi in moly_insert_indices:
            actual_center_points_sorted.insert(mi, (14540, -10830))
            actual_fov_names.insert(mi, 'MoQC')

        # easiest case: the center points and FOV names should be sorted
        if randomize_setting == ['N', 'N']:
            assert center_points == actual_center_points_sorted
            assert fov_names == actual_fov_names

        # if only the second ROI is randomized
        elif randomize_setting == ['N', 'Y']:
            # ensure the ROI 1 center points and FOV names are the same for both sorted and random
            assert center_points[:roi_1_end_pos] == actual_center_points_sorted[:roi_1_end_pos]
            assert fov_names[:roi_1_end_pos] == actual_fov_names[:roi_1_end_pos]

            # ensure the random center points and fov names for ROI 2 contain the same elements
            # as its sorted version
            misc_utils.verify_same_elements(
                computed_center_points=center_points[roi_1_end_pos:],
                actual_center_points=actual_center_points_sorted[roi_1_end_pos:]
            )
            misc_utils.verify_same_elements(
                computed_fov_names=fov_names[roi_1_end_pos:],
                actual_fov_names=actual_fov_names[roi_1_end_pos:]
            )

            # however, ROI 2 sorted entries should NOT equal ROI 2 random entries
            assert center_points[roi_1_end_pos:] != actual_center_points_sorted[roi_1_end_pos:]
            assert fov_names[roi_1_end_pos:] != actual_fov_names[roi_1_end_pos:]
        # if both fovs are randomized
        elif randomize_setting == ['Y', 'Y']:
            # ensure the random center points and fov names for ROI 1 contain the same elements
            # as its sorted version
            misc_utils.verify_same_elements(
                computed_center_points=center_points[:roi_1_end_pos],
                actual_center_points=actual_center_points_sorted[:roi_1_end_pos]
            )
            misc_utils.verify_same_elements(
                computed_fov_names=fov_names[:roi_1_end_pos],
                actual_fov_names=actual_fov_names[:roi_1_end_pos]
            )

            # however, ROI 1 sorted entries should NOT equal ROI 1 random entries
            assert center_points[:roi_1_end_pos] != actual_center_points_sorted[:roi_1_end_pos]
            assert fov_names[:roi_1_end_pos] != actual_fov_names[:roi_1_end_pos]

            # ensure the random center points for ROI 2 contain the same elements
            # as its sorted version
            misc_utils.verify_same_elements(
                computed_center_points=center_points[roi_1_end_pos:],
                actual_center_points=actual_center_points_sorted[roi_1_end_pos:]
            )
            misc_utils.verify_same_elements(
                computed_fov_names=fov_names[roi_1_end_pos:],
                actual_fov_names=actual_fov_names[roi_1_end_pos:]
            )

            # however, ROI 2 sorted entries should NOT equal ROI 2 random entries
            assert center_points[roi_1_end_pos:] != actual_center_points_sorted[roi_1_end_pos:]
            assert fov_names[roi_1_end_pos:] != actual_fov_names[roi_1_end_pos:]


@parametrize_with_cases('top_left, top_right, bottom_left, bottom_right',
                        cases=test_cases.ValidateRhombusCoordsCases)
def test_validate_tma_corners(top_left, top_right, bottom_left, bottom_right):
    tiling_utils.validate_tma_corners(top_left, top_right, bottom_left, bottom_right)


@parametrize('extra_coords,extra_names', [param([(1, 2)], ["TheSecondFOV"], marks=value_err),
                                          param([], [])])
@parametrize('num_row,num_col', [param(2, 3, marks=value_err), param(3, 2, marks=value_err),
                                 param(3, 4)])
@parametrize('tma_corners_file', [param('bad_path.json', marks=file_missing_err),
                                  param('sample_tma_corners.json')])
@parametrize_with_cases('coords, actual_pairs', cases=test_cases.RhombusCoordInputCases)
def test_generate_tma_fov_list(tma_corners_file, extra_coords, extra_names, num_row, num_col,
                               coords, actual_pairs):
    # extract the coordinates
    top_left, top_right, bottom_left, bottom_right = coords

    # generate a sample FOVs list
    # NOTE: extra_coords and extra_names are used to ensure failures
    # if the TMA spec file does not have 4 FOVS
    fov_coords = [astuple(top_left), astuple(top_right),
                  astuple(bottom_left), astuple(bottom_right)] + extra_coords
    fov_names = ["TheFirstFOV"] * 4 + extra_names

    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=fov_coords,
        fov_names=fov_names,
        fov_sizes=[5] * len(fov_coords)
    )

    with tempfile.TemporaryDirectory() as td:
        # save sample FOVs list
        sample_tma_corners_path = os.path.join(td, 'sample_tma_corners.json')
        with open(sample_tma_corners_path, 'w') as sfl:
            json.dump(sample_fovs_list, sfl)

        # NOTE: we leave the coordinate validation tests for test_validate_tma_corners

        # create the FOV regions
        fov_regions = tiling_utils.generate_tma_fov_list(
            os.path.join(td, tma_corners_file), num_row, num_col
        )

        # assert the correct number of fovs were created
        assert len(fov_regions) == num_row * num_col

        # get the list of fov names
        fov_names = list(fov_regions.keys())

        # specific tests for the corners: assert they are named correctly
        # NOTE: because of slanting, the coords may not match the originals in sample_fovs_list
        # we leave test_generate_x_y_fov_pairs_rhombus to test the coord assignment
        top_left_fov = fov_names[0]
        assert top_left_fov == 'R1C1'

        top_right_fov = fov_names[num_row * num_col - num_row]
        assert top_right_fov == 'R1C%d' % num_col

        bottom_left_fov = fov_names[num_row - 1]
        assert bottom_left_fov == 'R%dC1' % num_row

        bottom_right_fov = fov_names[(num_row * num_col) - 1]
        assert bottom_right_fov == 'R%dC%d' % (num_row, num_col)

        # now assert all the FOVs in between are named correctly and in the right order
        for i, fov in enumerate(fov_regions.keys()):
            row_ind = (i % num_row) + 1
            col_ind = int(i / num_row) + 1

            assert fov == 'R%dC%d' % (row_ind, col_ind)


def test_rename_duplicate_fovs():
    # define a sample set of FOVs
    fov_coords = [(100 + 100 * i, 100 + 100 * j) for i in np.arange(4) for j in np.arange(3)]
    fov_names = ["R%dC%d" % (i, j) for i in np.arange(1, 5) for j in np.arange(1, 4)]
    fov_sizes = [1000] * 12
    fov_list = test_utils.generate_sample_fovs_list(fov_coords, fov_names, fov_sizes)

    # no duplicate FOV names identified, no names should be changed
    fov_list_no_dup = tiling_utils.rename_duplicate_fovs(fov_list)
    assert [fov['name'] for fov in fov_list_no_dup['fovs']] == fov_names

    # rename R2C2 and R2C3 as R2C1 to create one set of duplicates
    fov_list['fovs'][4]['name'] = 'R2C1'
    fov_list['fovs'][5]['name'] = 'R2C1'
    fov_names[4] = 'R2C1_duplicate1'
    fov_names[5] = 'R2C1_duplicate2'

    fov_list_one_dup = tiling_utils.rename_duplicate_fovs(fov_list)
    assert [fov['name'] for fov in fov_list_one_dup['fovs']] == fov_names

    # rename R3C3, R4C1, and R4C2 as R3C2 to create another set of duplicates of differing size
    fov_list['fovs'][8]['name'] = 'R3C2'
    fov_list['fovs'][9]['name'] = 'R3C2'
    fov_list['fovs'][10]['name'] = 'R3C2'
    fov_names[8] = 'R3C2_duplicate1'
    fov_names[9] = 'R3C2_duplicate2'
    fov_names[10] = 'R3C2_duplicate3'

    fov_list_mult_dup = tiling_utils.rename_duplicate_fovs(fov_list)
    assert [fov['name'] for fov in fov_list_mult_dup['fovs']] == fov_names


def test_convert_stage_to_optical():
    # just need to test it gets the right values for one coordinate in microns
    sample_coord = (25000, 35000)

    # also need a sample set of co-registration params
    sample_coreg_params = {
        'STAGE_TO_OPTICAL_X_MULTIPLIER': 10,
        'STAGE_TO_OPTICAL_X_OFFSET': 1,
        'STAGE_TO_OPTICAL_Y_MULTIPLIER': 20,
        'STAGE_TO_OPTICAL_Y_OFFSET': -4
    }

    new_coord = tiling_utils.convert_stage_to_optical(sample_coord, sample_coreg_params)

    assert new_coord == (620, 257)


def test_assign_closest_fovs():
    # define the coordinates and fov names generated from the fov script
    # note that we intentionally define more auto fovs than manual fovs
    # to test that not all auto fovs necessarily get mapped to
    auto_coords = [(0, 0), (0, 5000), (0, 10000), (10000, 0), (10000, 5000), (10000, 10000),
                   (15000, 10000), (15000, 15000)]
    auto_fov_names = ['row%d_col%d' % (x, y) for (x, y) in auto_coords]

    # generate the list of automatically-generated fovs
    auto_sample_fovs = dict(zip(auto_fov_names, auto_coords))

    # define the coordinates and fov names proposed by the user
    manual_coords = [(0, 2500), (5000, 2500), (5000, 5000), (7500, 5000), (10000, 2500)]
    manual_fov_names = ['row%d_col%d' % (x, y) for (x, y) in manual_coords]

    # generate the list of manual fovs
    manual_sample_fovs = test_utils.generate_sample_fovs_list(
        manual_coords, manual_fov_names, fov_sizes=[5] * len(manual_coords)
    )

    # generate the mapping from manual to automatically-generated
    manual_to_auto_map, manual_auto_dist = \
        tiling_utils.assign_closest_fovs(
            manual_sample_fovs, auto_sample_fovs
        )

    # assert the mapping is correct, this covers 2 test cases:
    # 1. Not all auto fovs (ex. row0_col10000, row_10000_col10000) will be mapped to
    # 2. Multiple manual fovs can map to one auto fov (ex. row0_col25 and row50_col25 to row0_col0)
    # NOTE: if tied (ex. row0_col2500 to row0_col0 or row0_col5000), get the first alphabetically
    actual_map = {
        'row0_col2500': 'row0_col0',
        'row5000_col2500': 'row0_col0',
        'row5000_col5000': 'row0_col5000',
        'row7500_col5000': 'row10000_col5000',
        'row10000_col2500': 'row10000_col0'
    }

    assert manual_to_auto_map == actual_map

    # define the actual distance table, assert manual_auto_dist returned is correct
    actual_dist = np.linalg.norm(
        np.array(manual_coords)[:, np.newaxis] - np.array(auto_coords), axis=2
    )

    assert np.all(actual_dist == manual_auto_dist.values)


def test_generate_fov_circles():
    sample_manual_fovs_info = {
        'row0_col25': (0, 25),
        'row50_col25': (50, 25),
        'row50_col50': (50, 50),
        'row75_col50': (75, 50),
        'row100_col25': (100, 25)
    }

    sample_auto_fovs_info = {
        'row0_col0': (0, 0),
        'row0_col50': (0, 50),
        'row0_col100': (0, 100),
        'row100_col0': (100, 0),
        'row100_col50': (100, 50),
        'row100_col100': (100, 100)
    }

    # define the sample slide image
    sample_slide_img = np.full((200, 200, 3), 255)

    # draw the circles
    sample_slide_img = tiling_utils.generate_fov_circles(
        sample_manual_fovs_info, sample_auto_fovs_info, 'row0_col25', 'row0_col0',
        sample_slide_img, draw_radius=1
    )

    # assert the centroids are correct and they are filled in
    for pti in sample_manual_fovs_info:
        x, y = sample_manual_fovs_info[pti]

        # dark red if row0_col25, else bright red
        if pti == 'row0_col25':
            assert np.all(sample_slide_img[x, y, :] == np.array([210, 37, 37]))
        else:
            assert np.all(sample_slide_img[x, y, :] == np.array([255, 133, 133]))

    # same for the auto annotations
    for ati in sample_auto_fovs_info:
        x, y = sample_auto_fovs_info[ati]

        # dark blue if row0_col0, else bright blue
        if ati == 'row0_col0':
            assert np.all(sample_slide_img[x, y, :] == np.array([50, 115, 229]))
        else:
            assert np.all(sample_slide_img[x, y, :] == np.array([162, 197, 255]))


def test_update_mapping_display():
    manual_to_auto_map = {
        'row0_col25': 'row10_col10',
        'row50_col25': 'row10_col10',
        'row50_col50': 'row40_col10',
        'row75_col50': 'row40_col40'
    }

    manual_coords = {
        'row0_col25': (0, 25),
        'row50_col25': (50, 25),
        'row50_col50': (50, 50),
        'row75_col50': (75, 50),
    }

    auto_coords = {
        'row10_col10': (10, 10),
        'row10_col40': (10, 40),
        'row40_col10': (40, 10),
        'row40_col40': (40, 40)
    }

    # define a sample slide image
    slide_img = np.zeros((200, 200, 3))

    # test 1: select a new manual FOV that maps to the same auto FOV as the previous one
    change = {
        'old': 'row50_col25',
        'new': 'row0_col25'
    }

    # draw the old highlighted pair on the slide (first manual, then auto)
    # assume radius of 1 for all tests
    slide_img[50, 25, :] = [210, 37, 37]
    slide_img[10, 10, :] = [50, 115, 229]

    # draw the old non-highlighted pairs on the slide (first manual, then auto)
    for x, y in zip([0, 50, 75], [25, 50, 50]):
        slide_img[x, y, :] = [255, 133, 133]

    for x, y in zip([10, 40, 40], [40, 10, 40]):
        slide_img[x, y, :] = [162, 197, 255]

    # define a dummy automatic FOV scroller
    w_auto = widgets.Dropdown(
        options=[afi for afi in sorted(list(auto_coords.keys()))],
        value=manual_to_auto_map[change['old']]
    )

    # generate the new slide image
    new_slide_img = tiling_utils.update_mapping_display(
        change,
        w_auto,
        manual_to_auto_map,
        manual_coords,
        auto_coords,
        slide_img,
        draw_radius=1
    )

    # assert w_auto has been set correctly
    assert w_auto.value == 'row10_col10'

    # assert the new pairs are highlighted correctly (first manual, then auto)
    assert np.all(new_slide_img[0, 25, :] == [210, 37, 37])
    assert np.all(new_slide_img[10, 10, :] == [50, 115, 229])

    # assert all the others aren't highlighted (first manual, then auto)
    for x, y in zip([50, 50, 75], [25, 25, 50]):
        assert np.all(slide_img[x, y, :] == [255, 133, 133])

    for x, y in zip([10, 40, 40], [40, 10, 40]):
        assert np.all(slide_img[x, y, :] == [162, 197, 255])

    # test 2: select a new manual FOV that maps to a different auto FOV as the previous one
    change = {
        'old': 'row0_col25',
        'new': 'row50_col50'
    }

    # generate the new slide image
    new_slide_img = tiling_utils.update_mapping_display(
        change,
        w_auto,
        manual_to_auto_map,
        manual_coords,
        auto_coords,
        new_slide_img,
        draw_radius=1
    )

    # assert w_auto has been set correctly
    assert w_auto.value == 'row40_col10'

    # assert the new pairs are highlighted correctly (first manual, then auto)
    assert np.all(new_slide_img[50, 50, :] == [210, 37, 37])
    assert np.all(new_slide_img[40, 10, :] == [50, 115, 229])

    # assert all the others aren't highlighted (first manual, then auto)
    for x, y in zip([0, 50, 75], [25, 25, 50]):
        assert np.all(new_slide_img[x, y, :] == [255, 133, 133])

    for x, y in zip([10, 10, 40], [10, 40, 40]):
        assert np.all(new_slide_img[x, y, :] == [162, 197, 255])


# NOTE: we allow test_generate_validation_annot to run the annotation tests
# only the mapping update and the visualization updates are tested here
def test_remap_manual_to_auto_display():
    manual_to_auto_map = {
        'row0_col0': 'row3_col3',
        'row0_col5': 'row3_col3',
        'row5_col0': 'row6_col3',
        'row5_col5': 'row6_col6'
    }

    manual_coords = {
        'row0_col0': (0, 0),
        'row0_col5': (0, 5),
        'row5_col0': (5, 0),
        'row5_col5': (5, 5)
    }

    auto_coords = {
        'row3_col3': (3, 3),
        'row3_col6': (3, 6),
        'row6_col3': (6, 3),
        'row6_col6': (6, 6)
    }

    manual_auto_dist = np.zeros((4, 4))
    for i, mc in enumerate(manual_coords.values()):
        for j, ac in enumerate(auto_coords.values()):
            manual_auto_dist[i, j] = np.linalg.norm(np.array(mc) - np.array(ac))

    manual_auto_dist = pd.DataFrame(
        manual_auto_dist,
        index=list(manual_coords.keys()),
        columns=list(auto_coords.keys())
    )

    # define a sample slide image
    slide_img = np.zeros((200, 200, 3))

    # define a dummy manual FOV scroller
    w_man = widgets.Dropdown(
        options=[afi for afi in sorted(list(manual_coords.keys()))],
        value='row0_col0'
    )

    # define the old and the new auto FOV to map to
    change = {
        'old': 'row3_col3',
        'new': 'row3_col6'
    }

    # draw the old highlighted pair on the slide (first manual, then auto)
    # assume radius of 1 for all tests
    slide_img[0, 0, :] = [210, 37, 37]
    slide_img[3, 3, :] = [50, 115, 229]

    # draw the old non-highlighted pairs on the slide (first manual, then auto)
    for x, y in zip([0, 5, 5], [5, 0, 5]):
        slide_img[x, y, :] = [255, 133, 133]

    for x, y in zip([3, 6, 6], [6, 3, 6]):
        slide_img[x, y, :] = [162, 197, 255]

    # generate the new slide image, use default annotation params as we won't be testing that here
    new_slide_img, _ = tiling_utils.remap_manual_to_auto_display(
        change,
        w_man,
        manual_to_auto_map,
        manual_auto_dist,
        auto_coords,
        slide_img,
        draw_radius=1
    )

    # assert the new mapping has been made
    assert manual_to_auto_map['row0_col0'] == 'row3_col6'

    # assert the new pairs are highlighted correctly (first manual, then auto)
    assert np.all(new_slide_img[0, 0, :] == [210, 37, 37])
    assert np.all(new_slide_img[3, 6, :] == [50, 115, 229])

    # assert all the others aren't highlighted (first manual, then auto)
    for x, y in zip([0, 5, 5], [5, 0, 5]):
        assert np.all(new_slide_img[x, y, :] == [255, 133, 133])

    for x, y in zip([3, 6, 6], [3, 3, 6]):
        assert np.all(new_slide_img[x, y, :] == [162, 197, 255])


# NOTE: it won't be possible to test the exact datetime the mapping was saved at
@parametrize('annot', [None, plt.annotate('Mapping saved at', (10, 10))])
def test_save_json(annot):
    # define the save annotation status
    save_ann = {'annotation': annot}

    with tempfile.TemporaryDirectory() as td:
        # write a sample json and update the annotation
        tiling_utils.save_json(
            test_cases._ANNOT_SAMPLE_MAPPING,
            save_ann,
            os.path.join(td, 'sample_mapping.json')
        )

        # assert the mapping file was saved
        assert os.path.exists(os.path.join(td, 'sample_mapping.json'))

        # assert the annotation got updated
        assert save_ann['annotation'] is not None


@parametrize_with_cases('manual_to_auto_map, manual_auto_dist, actual_bad_dist_list',
                        cases=test_cases.MappingDistanceCases)
def test_find_manual_auto_invalid_dist(manual_to_auto_map, manual_auto_dist, actual_bad_dist_list):
    generated_bad_dist_list = tiling_utils.find_manual_auto_invalid_dist(
        manual_to_auto_map, manual_auto_dist, dist_threshold=50
    )

    assert generated_bad_dist_list == actual_bad_dist_list


@parametrize_with_cases('manual_to_auto_map, actual_duplicate_list',
                        cases=test_cases.MappingDuplicateCases)
def test_find_duplicate_auto_mappings(manual_to_auto_map, actual_duplicate_list):
    generated_duplicate_list = tiling_utils.find_duplicate_auto_mappings(manual_to_auto_map)

    assert generated_duplicate_list == actual_duplicate_list


@parametrize_with_cases('manual_to_auto_map, actual_mismatch_list',
                        cases=test_cases.MappingMismatchCases)
def test_find_manual_auto_name_mismatches(manual_to_auto_map, actual_mismatch_list):
    generated_mismatch_list = tiling_utils.find_manual_auto_name_mismatches(manual_to_auto_map)

    assert generated_mismatch_list == actual_mismatch_list


@parametrize('check_dist', [None, 50])
@parametrize('check_duplicates', [False, True])
@parametrize('check_mismatches', [False, True])
def test_generate_validation_annot(check_dist, check_duplicates, check_mismatches):
    generated_annot = tiling_utils.generate_validation_annot(
        test_cases._ANNOT_SAMPLE_MAPPING, test_cases._ANNOT_SAMPLE_DIST,
        check_dist, check_duplicates, check_mismatches
    )

    actual_annot = test_cases.generate_sample_annot(check_dist, check_duplicates, check_mismatches)

    assert generated_annot == actual_annot


def test_generate_fov_rectangle():
    # define the centroid of the rectangle and its size
    fov_coord = (10000, 10000)
    fov_size = 2500

    # generate a FOV to draw a rectangle for
    fov_info = test_utils.generate_sample_fov_tiling_entry(
        coord=fov_coord, name='Sample_ROI_R1C1', size=fov_size
    )

    # define a sample mapping from region name to RGBA color
    region_colors = {
        'Sample_ROI': (1.0, 0.0, 0.0, 1.0),
        'Other_ROI': (0.0, 0.0, 1.0, 1.0)
    }

    # hard-code a dummy set of co-registration parameters
    stage_optical_coreg_params = {
        'STAGE_TO_OPTICAL_X_MULTIPLIER': 25,
        'STAGE_TO_OPTICAL_X_OFFSET': 750,
        'STAGE_TO_OPTICAL_Y_MULTIPLIER': -20,
        'STAGE_TO_OPTICAL_Y_OFFSET': -500
    }

    # create a sample axis to draw on
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # draw the rectangle
    dr = tiling_utils.generate_fov_rectangle(
        fov_info, region_colors, stage_optical_coreg_params, ax
    )

    # assert the coordinate of the upper-left corner matches
    assert dr.rect.get_xy() == (9783, 18961)

    # assert the height and width are computed correctly
    # NOTE: actual co-registration parameters mean height and width are much closer in magnitude
    assert dr.rect.get_height() == 51
    assert dr.rect.get_width() == 62

    # assert the edge color is set correctly
    assert dr.rect.get_edgecolor() == region_colors['Sample_ROI']

    # assert the id of the rectangle is set to the name of the FOV
    assert dr.id_val == 'Sample_ROI_R1C1'

    # assert the rectangle is not pressed
    assert not dr.press


# @parametrize_with_cases('rectangle_list', test_cases.RectangleCases)
@parametrize('randomize', [False, True])
@parametrize('moly', [False, True])
def test_delete_tiled_region_fovs(randomize, moly):
    # define a 4 FOVs for region 1 and 6 for region 2
    fov_names_one = [
        'Sample_ROI1_R%dC%d' % (i, j) for i in np.arange(1, 3) for j in np.arange(1, 3)
    ]
    fov_names_two = [
        'Sample_ROI2_R%dC%d' % (i, j) for i in np.arange(1, 3) for j in np.arange(1, 4)
    ]

    # shuffle if randomize is set
    if randomize:
        fov_names_one = shuffle(fov_names_one)
        fov_names_two = shuffle(fov_names_two)

    fov_names = fov_names_one + fov_names_two

    # NOTE: for testing deletion, actual coordinate values and sizes are not important
    # we leave test_generate_fov_rectangle to verify that co-registration works properly
    fov_coords = [tuple(np.random.rand(2)) for i in np.arange(len(fov_names))]
    fov_sizes = [5000] * len(fov_names)

    # define a sample axis to draw the rectangles on
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # we'll randomly select a set of FOVs to delete, store which ones for testing
    fovs_delete = []

    # generate the set of rectangles for each FOV
    rectangles = {}
    for fov_name, fov_coord, fov_size in zip(fov_names, fov_coords, fov_sizes):
        # make sure to skip Moly points!
        # NOTE: just assume no co-registration is required for this test
        if fov_name != 'MoQC':
            dr = tiling_utils.FOVRectangle(
                fov_coord, fov_size, fov_size, (1.0, 0.0, 0.0, 1.0), fov_name, ax
            )

            # decide whether we'll remove this FOV or not with a 25% chance
            keep = np.random.rand(1)[0]
            if keep < 0.25:
                dr.rect.set_linewidth(5)
                fovs_delete.append(fov_name)

            rectangles[fov_name] = dr

    # insert moly points if specified
    if moly:
        moly_index = 3
        while moly_index < len(fov_names):
            fov_names.insert(moly_index, 'MoQC')
            fov_coords.insert(moly_index, (-1000, -1000))
            fov_sizes.insert(moly_index, 1000)

            moly_index += 4

    # generate the run JSON for these names
    tiled_region_fovs = test_utils.generate_sample_fovs_list(fov_coords, fov_names, fov_sizes)

    # run the deletion process
    tiling_utils.delete_tiled_region_fovs(rectangles, tiled_region_fovs)

    # generate the set of FOVs that still appear
    # NOTE: ignore Moly points
    rectangle_names_list = list(rectangles.keys())
    tiled_region_names_list = [
        fov['name'] for fov in tiled_region_fovs['fovs'] if fov['name'] != 'MoQC'
    ]

    # these names should be the same
    misc_utils.verify_same_elements(
        fov_names_rectangles=rectangle_names_list,
        fov_names_run_json=tiled_region_names_list
    )

    # ...and these should include all the FOVs not deleted
    # NOTE: ignore Moly points
    fovs_not_deleted = [
        fov for fov in list(set(fov_names) - set(fovs_delete)) if fov != 'MoQC'
    ]
    misc_utils.verify_same_elements(
        fov_names_rectangles=rectangle_names_list,
        fovs_to_keep=fovs_not_deleted
    )
    misc_utils.verify_same_elements(
        fov_names_run_json=tiled_region_names_list,
        fovs_to_keep=fovs_not_deleted
    )

    # finally, assert no line width of 5 exists in rectangles
    # an extra test that all FOVs marked for removal were actually deleted
    linewidths = np.array([dr.rect.get_linewidth() for dr in rectangles.values()])
    assert np.all(linewidths == 1)


# # NOTE: this only tests if the rectangles are drawn correctly
# # unfortunately, there isn't a way to programmatically test the click of a button on Python's side
# # see: https://github.com/jupyter-widgets/ipywidgets/issues/1896
# @parametrize('num_regions', [1, 2, 3])
# @parametrize('randomize', [False, True])
# @parametrize('moly', [False, True])
# def test_tiled_region_interactive_remap(num_regions, randomize, moly):
#     # define the set of tiled region params, this will contain just the top-left corners
#     sample_tiled_region_params = {
#         "exportDateTime": "2021-03-12T19:02:37.920Z",
#         "fovFormatVersion": "1.5",
#         "fovs": [],
#         "region_params": []
#     }

#     # iteratively generate the top-left FOV entry for each, as well as the region_params
#     for roi in num_regions:
#         pass

#     # define the final list of FOVs to use
#     sample_tiled_region_fovs = {
#         "exportDateTime": "2021-03-12T19:02:37.920Z",
#         "fovFormatVersion": "1.5",
#         "fovs": []
#     }

#     # iteratively generate a FOV list for each region
#     for roi in num_regions:
#         # define the name of this region, this will prefix the FOV names
#         roi_name = 'Sample_ROI%d' % roi

#         # define the number of rows and columns for this region
#         num_row_fovs = 2 * roi
#         num_col_fovs = 3 * roi

#         # define the top-left centroid of this set of FOVs
#         fov_row_start = 10000 * roi
#         fov_col_start = 20000 * roi

#         # define the size of these FOVs, note that this will be consistent due to the way
#         # fovSizeMicrons is computed across all the FOVs in an ROI
#         fov_size = 2500 * roi

#         # define all the coordinates to use
#         fov_coords = [
#             (fov_row_start + fov_size * (i + 1), fov_col_start + fov_size * (j + 1))
#             for i in np.arange(num_row_fovs)
#             for j in np.arange(num_col_fovs)
#         ]

#         # define the corresponding FOV names
#         fov_names = [
#             '%s_R%dC%d' % (roi_name, i + 1, j + 1)
#             for i in np.arange(num_row_fovs)
#             for j in np.arange(num_col_fovs)
#         ]

#         fov_coords = [(1000 * (i + 1), 2000 * ())]
#         start_coords = (1000 * roi, 2000 * roi)

#     # retrieve the colormap used by this visualization for testing
#     cmap = plt.get_cmap('tab20')


# NOTE: this only tests if the visualization runs with valid parameters
# previous test functions check interactive functionality
def test_tma_interactive_remap():
    with tempfile.TemporaryDirectory() as temp_dir:
        # make a dummy toffy directory and a dummy templates directory where the code is run
        os.mkdir(os.path.join(temp_dir, 'toffy'))
        os.mkdir(os.path.join(temp_dir, 'templates'))

        # change working directory to templates to simulate actual co-registration run
        os.chdir(os.path.join(temp_dir, 'templates'))

        # define sample data for each parameter
        sample_manual_fovs = {
            'fovs': [
                {
                    'name': 'R1C1',
                    'centerPointMicrons': {
                        'x': 100,
                        'y': 100
                    }
                },
                {
                    'name': 'R2C2',
                    'centerPointMicrons': {
                        'x': 200,
                        'y': 200
                    }
                }
            ]
        }

        sample_auto_fovs = {
            'R1C1': (125, 125),
            'R1C2': (125, 225),
            'R2C1': (225, 125),
            'R2C2': (225, 225)
        }

        sample_slide_img = np.zeros((1024, 1024, 3))

        mapping_path = os.path.join('..', 'toffy', 'mapping.json')

        # error check: directory path to mapping needs to be valid
        with pytest.raises(FileNotFoundError):
            tiling_utils.tma_interactive_remap(
                sample_manual_fovs, sample_auto_fovs, sample_slide_img,
                'bad/bad/bad_mapping_path.json'
            )

        # error check: check_dist needs to be numeric
        with pytest.raises(ValueError):
            tiling_utils.tma_interactive_remap(
                sample_manual_fovs, sample_auto_fovs, sample_slide_img,
                mapping_path, check_dist='bad'
            )

        # error check: check_dist needs to be greater than 0
        with pytest.raises(ValueError):
            tiling_utils.tma_interactive_remap(
                sample_manual_fovs, sample_auto_fovs, sample_slide_img,
                mapping_path, check_dist=0
            )

        # error check: check_duplicates needs to be boolean
        with pytest.raises(ValueError):
            tiling_utils.tma_interactive_remap(
                sample_manual_fovs, sample_auto_fovs, sample_slide_img,
                mapping_path, check_duplicates='bad'
            )

        # error check: check_mismatches needs to be boolean
        with pytest.raises(ValueError):
            tiling_utils.tma_interactive_remap(
                sample_manual_fovs, sample_auto_fovs, sample_slide_img,
                mapping_path, check_mismatches='bad'
            )

        # error check: no co-registration params specified
        with pytest.raises(FileNotFoundError):
            tiling_utils.tma_interactive_remap(
                sample_manual_fovs, sample_auto_fovs, sample_slide_img, mapping_path
            )

        # generate sample co-registration params in toffy
        sample_coreg_params = {
            'coreg_params': [
                {
                    'STAGE_TO_OPTICAL_X_MULTIPLIER': 10,
                    'STAGE_TO_OPTICAL_X_OFFSET': 1,
                    'STAGE_TO_OPTICAL_Y_MULTIPLIER': 20,
                    'STAGE_TO_OPTICAL_Y_OFFSET': -4
                }
            ]
        }

        with open(os.path.join('..', 'toffy', 'coreg_params.json'), 'w') as cp:
            json.dump(sample_coreg_params, cp)

        # this should now run
        tiling_utils.tma_interactive_remap(
            sample_manual_fovs, sample_auto_fovs, sample_slide_img, mapping_path
        )


@parametrize('randomize_setting', [False, True])
@parametrize('moly_insert, moly_interval', test_cases._REMAP_MOLY_INTERVAL_CASES)
@parametrize('moly_path', [param('bad_moly_point.json', marks=file_missing_err),
                           param('sample_moly_point.json')])
def test_remap_and_reorder_fovs(moly_path, randomize_setting, moly_insert, moly_interval):
    # define the sample Moly point
    sample_moly_point = test_utils.generate_sample_fov_tiling_entry(
        coord=(14540, -10830), name="MoQC", size=10000
    )

    # save the Moly point
    with tempfile.TemporaryDirectory() as td:
        moly_point_path = os.path.join(td, 'sample_moly_point.json')

        with open(moly_point_path, 'w') as smp:
            json.dump(sample_moly_point, smp)

        # define the coordinates and fov names manual by the user
        manual_coords = [(0, 25), (50, 25), (50, 50), (75, 50), (100, 25), (100, 75)]
        manual_fov_names = ['row%d_col%d' % (x, y) for (x, y) in manual_coords]

        # generate the list of manual fovs
        manual_sample_fovs = test_utils.generate_sample_fovs_list(
            manual_coords, manual_fov_names, fov_sizes=[5] * len(manual_coords)
        )

        # define a sample mapping
        sample_mapping = {
            'row0_col25': 'row0_col0',
            'row50_col25': 'row0_col25',
            'row50_col50': 'row0_col100',
            'row75_col50': 'row100_col100',
            'row100_col25': 'row100_col0',
            'row100_col75': 'row100_col75'
        }

        # copy the data so it doesn't overwrite manual_sample_fovs
        manual_sample_fovs_copy = copy.deepcopy(manual_sample_fovs)

        # add id, name, and status
        manual_sample_fovs_copy['id'] = -1
        manual_sample_fovs_copy['name'] = 'test'
        manual_sample_fovs_copy['status'] = 'all_systems_go'

        # remap the FOVs
        remapped_sample_fovs = tiling_utils.remap_and_reorder_fovs(
            manual_sample_fovs_copy, sample_mapping, os.path.join(td, moly_path),
            randomize_setting, moly_insert, moly_interval
        )

        # assert id, name, and status are the same
        assert remapped_sample_fovs['id'] == manual_sample_fovs_copy['id']
        assert remapped_sample_fovs['name'] == manual_sample_fovs_copy['name']
        assert remapped_sample_fovs['status'] == manual_sample_fovs_copy['status']

        # assert the same FOVs in the manual-to-auto map (sample_mapping)
        # appear in remapped sample FOVs after the remapping process
        misc_utils.verify_same_elements(
            remapped_fov_names=[fov['name'] for fov in remapped_sample_fovs['fovs']
                                if fov['name'] != 'MoQC'],
            fovs_in_mapping=list(sample_mapping.values())
        )

        # assert the mapping was done correctly
        scrambled_names = [fov['name'] for fov in remapped_sample_fovs['fovs']]
        for fov in manual_sample_fovs['fovs']:
            mapped_name = sample_mapping[fov['name']]
            assert mapped_name in scrambled_names

        # assert the same FOV coords are contained in remapped_sample_fovs as manual_sample_fovs
        scrambled_coords = [(fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
                            for fov in remapped_sample_fovs['fovs'] if fov['name'] != 'MoQC']
        misc_utils.verify_same_elements(
            scrambled_fov_coords=scrambled_coords,
            actual_coords=manual_coords
        )

        # enforce order–or not–depending on if randomization is added or not
        if randomize_setting:
            assert scrambled_coords != manual_coords
        else:
            assert scrambled_coords == manual_coords

        # if Moly points will be inserted, assert they are in the right place at the right interval
        # otherwise, assert no Moly points appear
        if moly_insert:
            # assert the moly_indices are inserted at the correct locations
            moly_indices = np.arange(
                moly_interval, len(remapped_sample_fovs['fovs']), moly_interval + 1
            )
            for mi in moly_indices:
                assert remapped_sample_fovs['fovs'][mi]['name'] == 'MoQC'
        else:
            # assert no Moly points appear in the list of fovs
            fov_names = [remapped_sample_fovs['fovs'][i]['name']
                         for i in range(len(remapped_sample_fovs['fovs']))]
            assert 'MoQC' not in fov_names
