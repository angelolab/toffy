import copy
import ipywidgets as widgets
import json
import numpy as np
import os
import pytest
from pytest_cases import parametrize_with_cases
import random
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


@parametrize_with_cases('fov_coords, fov_names, user_inputs, base_param_values, full_param_set',
                        cases=test_cases.TiledRegionReadCases, glob='*_no_moly_param')
def test_read_tiled_region_inputs(monkeypatch, fov_coords, fov_names, user_inputs,
                                  base_param_values, full_param_set):
    # define a sample fovs list
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=fov_coords, fov_names=fov_names
    )

    # define sample region_params to read data into
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
@parametrize_with_cases('fov_coords, fov_names, user_inputs, base_param_values, full_param_set',
                        cases=test_cases.TiledRegionReadCases, glob='*_with_moly_param')
@parametrize('moly_interval_val', [0, 1])
def test_set_tiled_region_params(monkeypatch, region_corners_file, fov_coords, fov_names,
                                 user_inputs, base_param_values,
                                 full_param_set, moly_interval_val):
    # define a sample set of fovs
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=fov_coords, fov_names=fov_names
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

        # test the value of each tiling param for both regions
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
        top_left, top_right, bottom_left, bottom_right, 4, 3
    )

    assert pairs == actual_pairs


@parametrize_with_cases(
    'moly_path, moly_region_setting, moly_interval_setting, moly_interval_value, '
    'moly_insert_indices, fov_1_end_pos', cases=test_cases.TiledRegionMolySettingCases
)
@parametrize('randomize_setting', [['N', 'N'], ['N', 'Y'], ['Y', 'Y']])
def test_generate_tiled_region_fov_list(moly_path, moly_region_setting,
                                        moly_interval_setting, moly_interval_value,
                                        moly_insert_indices, fov_1_end_pos, randomize_setting):
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=[(0, 0), (100, 100)], fov_names=["TheFirstFOV", "TheSecondFOV"]
    )

    sample_region_inputs = {
        'region_start_x': [0, 50],
        'region_start_y': [100, 150],
        'fov_num_x': [2, 4],
        'fov_num_y': [4, 2],
        'x_fov_size': [5, 10],
        'y_fov_size': [10, 5],
        'region_rand': ['N', 'N']
    }

    sample_region_params = tiling_utils.generate_region_info(sample_region_inputs)

    sample_tiling_params = {
        'fovFormatVersion': '1.5',
        'fovs': sample_fovs_list['fovs'],
        'region_params': sample_region_params
    }

    with tempfile.TemporaryDirectory() as td:
        sample_moly_point = test_utils.generate_sample_fov_tiling_entry(
            coord=(14540, -10830), name="MoQC"
        )
        sample_moly_path = os.path.join(td, 'sample_moly_point.json')

        with open(sample_moly_path, 'w') as smp:
            json.dump(sample_moly_point, smp)

        sample_tiling_params['moly_region'] = moly_region_setting

        sample_tiling_params['region_params'][0]['region_rand'] = randomize_setting[0]
        sample_tiling_params['region_params'][1]['region_rand'] = randomize_setting[1]

        if moly_interval_setting:
            sample_tiling_params['moly_interval'] = moly_interval_value

        fov_regions = tiling_utils.generate_tiled_region_fov_list(
            sample_tiling_params, os.path.join(td, moly_path)
        )

        # assert none of the metadata keys explicitly added by set_tiling_params appear
        for k in ['region_params', 'moly_region', 'moly_interval']:
            assert k not in fov_regions

        # retrieve the center points
        center_points = [
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            for fov in fov_regions['fovs']
        ]

        # define the center points sorted
        actual_center_points_sorted = [
            (x, y) for x in np.arange(0, 10, 5) for y in list(reversed(np.arange(70, 110, 10)))
        ] + [
            (x, y) for x in np.arange(50, 90, 10) for y in list(reversed(np.arange(145, 155, 5)))
        ]

        for mi in moly_insert_indices:
            actual_center_points_sorted.insert(mi, (14540, -10830))

        # easiest case: the center points should be sorted
        if randomize_setting == ['N', 'N']:
            assert center_points == actual_center_points_sorted
        # if only the second fov is randomized
        elif randomize_setting == ['N', 'Y']:
            # ensure the fov 1 center points are the same for both sorted and random
            assert center_points[:fov_1_end_pos] == actual_center_points_sorted[:fov_1_end_pos]

            # ensure the random center points for fov 2 contain the same elements
            # as its sorted version
            misc_utils.verify_same_elements(
                computed_center_points=center_points[fov_1_end_pos:],
                actual_center_points=actual_center_points_sorted[fov_1_end_pos:]
            )

            # however, fov 2 sorted entries should NOT equal fov 2 random entries
            # NOTE: due to randomization, this test will fail once in a blue moon
            assert center_points[fov_1_end_pos:] != actual_center_points_sorted[fov_1_end_pos:]
        # if both fovs are randomized
        elif randomize_setting == ['Y', 'Y']:
            # ensure the random center points for fov 1 contain the same elements
            # as its sorted version
            misc_utils.verify_same_elements(
                computed_center_points=center_points[:fov_1_end_pos],
                actual_center_points=actual_center_points_sorted[:fov_1_end_pos]
            )

            # however, fov 1 sorted entries should NOT equal fov 1 random entries
            # NOTE: due to randomization, this test will fail once in a blue moon
            assert center_points[:fov_1_end_pos] != actual_center_points_sorted[:fov_1_end_pos]

            # ensure the random center points for fov 2 contain the same elements
            # as its sorted version
            misc_utils.verify_same_elements(
                computed_center_points=center_points[fov_1_end_pos:],
                actual_center_points=actual_center_points_sorted[fov_1_end_pos:]
            )

            # however, fov 2 sorted entries should NOT equal fov 2 random entries
            # NOTE: due to randomization, this test will fail once in a blue moon
            assert center_points[fov_1_end_pos:] != actual_center_points_sorted[fov_1_end_pos:]


@parametrize_with_cases('top_left, top_right, bottom_left, bottom_right',
                        cases=test_cases.ValidateRhombusCoordsCases)
def test_validate_tma_corners(top_left, top_right, bottom_left, bottom_right):
    tiling_utils.validate_tma_corners(top_left, top_right, bottom_left, bottom_right)


@parametrize('extra_coords,extra_names', [param([(1, 2)], ["TheSecondFOV"], marks=value_err),
                                          param([], [])])
@parametrize('num_x,num_y', [param(2, 3, marks=value_err), param(3, 2, marks=value_err),
                             param(4, 3)])
@parametrize('tma_corners_file', [param('bad_path.json', marks=file_missing_err),
                                  param('sample_tma_corners.json')])
@parametrize_with_cases('coords, actual_pairs', cases=test_cases.RhombusCoordInputCases)
def test_generate_tma_fov_list(tma_corners_file, extra_coords, extra_names, num_x, num_y,
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
        fov_names=fov_names
    )

    with tempfile.TemporaryDirectory() as td:
        # save sample FOVs list
        sample_tma_corners_path = os.path.join(td, 'sample_tma_corners.json')
        with open(sample_tma_corners_path, 'w') as sfl:
            json.dump(sample_fovs_list, sfl)

        # NOTE: we leave the coordinate validation tests for test_validate_tma_corners

        # create the FOV regions
        fov_regions = tiling_utils.generate_tma_fov_list(
            os.path.join(td, tma_corners_file), num_x, num_y
        )

        # assert the correct number of fovs were created
        assert len(fov_regions) == num_x * num_y

        # get the list of fov names
        fov_names = list(fov_regions.keys())

        # specific tests for the corners: assert they are named correctly
        # NOTE: because of slanting, the coords may not match the originals in sample_fovs_list
        # we leave test_generate_x_y_fov_pairs_rhombus to test the
        # correctness of the coord assignment
        top_left_fov = fov_names[0]
        assert top_left_fov == 'R1C1'

        top_right_fov = fov_names[num_x * num_y - num_y]
        assert top_right_fov == 'R1C%d' % num_x

        bottom_left_fov = fov_names[num_y - 1]
        assert bottom_left_fov == 'R%dC1' % num_y

        bottom_right_fov = fov_names[(num_x * num_y) - 1]
        assert bottom_right_fov == 'R%dC%d' % (num_y, num_x)

        # now assert all the FOVs in between are named correctly and in the right order
        for i, fov in enumerate(fov_regions.keys()):
            row_ind = (i % num_y) + 1
            col_ind = int(i / num_y) + 1

            assert fov == 'R%dC%d' % (row_ind, col_ind)


def test_convert_microns_to_pixels():
    # just need to test it gets the right values for one coordinate in microns
    sample_coord = (25000, 35000)
    new_coord = tiling_utils.convert_microns_to_pixels(sample_coord)

    assert new_coord == (612, 762)


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
        manual_coords, manual_fov_names
    )

    # generate the mapping from manual to automatically-generated
    manual_to_auto_map, manual_fovs_info, auto_fovs_info = \
        tiling_utils.assign_closest_fovs(
            manual_sample_fovs, auto_sample_fovs
        )

    # for each manual fov, ensure the centroids are the same in manual_fovs_info
    for fov in manual_sample_fovs['fovs']:
        manual_centroid = tiling_utils.convert_microns_to_pixels(
            tuple(fov['centerPointMicrons'].values())
        )

        assert manual_fovs_info[fov['name']] == manual_centroid

    # same for automatically-generated fovs
    for fov in auto_sample_fovs:
        auto_centroid = tiling_utils.convert_microns_to_pixels(
            auto_sample_fovs[fov]
        )

        assert auto_fovs_info[fov] == auto_centroid

    # assert the mapping is correct, this covers 2 other test cases:
    # 1. Not all auto fovs (ex. row150_col100 and row150_col150) will be mapped to
    # 2. Multiple manual fovs can map to one auto fov (ex. row0_col25 and row50_col25 to row0_col0)
    actual_map = {
        'row0_col2500': {
            'closest_auto_fov': 'row0_col5000',
            'distance': 36.0
        },
        'row5000_col2500': {
            'closest_auto_fov': 'row0_col5000',
            'distance': np.linalg.norm(
                np.array([1089, 471]) - np.array([1053, 398])  # micron to pixel converted coords
            )
        },
        'row5000_col5000': {
            'closest_auto_fov': 'row0_col5000',
            'distance': 73.0
        },
        'row7500_col5000': {
            'closest_auto_fov': 'row10000_col5000',
            'distance': 37.0
        },
        'row10000_col2500': {
            'closest_auto_fov': 'row10000_col5000',
            'distance': 36.0
        }
    }

    assert manual_to_auto_map == actual_map


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
        'row0_col25': {
            'closest_auto_fov': 'row0_col0',
            'distance': 25.0
        },
        'row50_col25': {
            'closest_auto_fov': 'row0_col0',
            'distance': np.linalg.norm(np.array([50, 25]) - np.array([0, 0]))
        },
        'row50_col50': {
            'closest_auto_fov': 'row0_col50',
            'distance': 50.0
        },
        'row75_col50': {
            'closest_auto_fov': 'row100_col50',
            'distance': np.linalg.norm(np.array([75, 50]) - np.array([100, 50]))
        },
        'row100_col25': {
            'closest_auto_fov': 'row100_col0',
            'distance': 25.0
        }
    }

    manual_coords = {
        'row0_col25': (0, 25),
        'row50_col25': (50, 25),
        'row50_col50': (50, 50),
        'row75_col50': (75, 50),
        'row100_col25': (100, 25)
    }

    # NOTE: we intentionally define more auto coords than usual, test should not fail
    auto_coords = {
        'row0_col0': (0, 0),
        'row0_col50': (0, 50),
        'row100_col0': (100, 0),
        'row100_col50': (100, 50),
        'row100_col75': (100, 75)
    }

    # define a sample slide image
    slide_img = np.zeros((200, 200, 3))

    # test 1: select a new manual FOV that maps to the same auto FOV as the previous one
    # define the change dict
    change = {
        'old': 'row50_col25',
        'new': 'row0_col25'
    }

    # draw the old highlighted pair on the slide (first manual, then auto)
    # assume radius of 1 for all tests
    slide_img[50, 25, :] = [210, 37, 37]
    slide_img[0, 0, :] = [50, 115, 229]

    # draw the old non-highlighted pairs on the slide (first manual, then auto)
    for x, y in zip([0, 50, 75, 100], [25, 50, 50, 25]):
        slide_img[x, y, :] = [255, 133, 133]

    for x, y in zip([0, 100, 100, 100], [50, 0, 50, 75]):
        slide_img[x, y, :] = [162, 197, 255]

    # define a dummy automatic FOV scroller
    w_auto = widgets.Dropdown(
        options=[afi for afi in sorted(list(auto_coords.keys()))],
        value=manual_to_auto_map[change['old']]['closest_auto_fov']
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

    # assert the new pairs are highlighted correctly (first manual, then auto)
    assert np.all(new_slide_img[0, 25, :] == [210, 37, 37])
    assert np.all(new_slide_img[0, 0, :] == [50, 115, 229])

    # assert all the others aren't highlighted (first manual, then auto)
    for x, y in zip([50, 50, 75, 100], [25, 50, 50, 25]):
        assert np.all(slide_img[x, y, :] == [255, 133, 133])

    for x, y in zip([0, 100, 100, 100], [50, 0, 50, 75]):
        assert np.all(slide_img[x, y, :] == [162, 197, 255])

    # test 2: select a new manual FOV that maps to a different auto FOV as the previous one
    # define the change dict
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

    # assert the new pairs are highlighted correctly (first manual, then auto)
    assert np.all(new_slide_img[50, 50, :] == [210, 37, 37])
    assert np.all(new_slide_img[0, 50, :] == [50, 115, 229])

    # assert all the others aren't highlighted (first manual, then auto)
    for x, y in zip([0, 50, 75, 100], [25, 25, 50, 25]):
        assert np.all(new_slide_img[x, y, :] == [255, 133, 133])

    for x, y in zip([0, 100, 100, 100], [0, 0, 50, 75]):
        assert np.all(new_slide_img[x, y, :] == [162, 197, 255])


# NOTE: we allow test_generate_validation_annot to run the annotation tests
# only the mapping update and the visualization updates are tested here
def test_remap_manual_to_auto_display():
    manual_to_auto_map = {
        'row0_col0': {
            'closest_auto_fov': 'row3_col3',
            'distance': np.linalg.norm(np.array([0, 0]) - np.array([3, 3]))
        },
        'row0_col5': {
            'closest_auto_fov': 'row3_col3',
            'distance': np.linalg.norm(np.array([0, 5]) - np.array([3, 3]))
        },
        'row5_col0': {
            'closest_auto_fov': 'row6_col3',
            'distance': np.linalg.norm(np.array([5, 0]) - np.array([6, 3]))
        },
        'row5_col5': {
            'closest_auto_fov': 'row6_col6',
            'distance': np.linalg.norm(np.array([5, 5]) - np.array([6, 6]))
        }
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
        manual_coords,
        auto_coords,
        slide_img,
        draw_radius=1
    )

    # assert the new mapping has been made with updated distance
    assert manual_to_auto_map['row0_col0']['closest_auto_fov'] == 'row3_col6'
    assert manual_to_auto_map['row0_col0']['distance'] == np.linalg.norm(
        np.array([0, 0]) - np.array([3, 6])
    )

    # assert the new pairs are highlighted correctly (first manual, then auto)
    assert np.all(new_slide_img[0, 0, :] == [210, 37, 37])
    assert np.all(new_slide_img[3, 6, :] == [50, 115, 229])

    # assert all the others aren't highlighted (first manual, then auto)
    for x, y in zip([0, 5, 5], [5, 0, 5]):
        assert np.all(new_slide_img[x, y, :] == [255, 133, 133])

    for x, y in zip([3, 6, 6], [3, 3, 6]):
        assert np.all(new_slide_img[x, y, :] == [162, 197, 255])


@parametrize_with_cases('manual_to_auto_map, actual_bad_dist_list',
                        cases=test_cases.MappingDistanceCases)
def test_find_manual_auto_invalid_dist(manual_to_auto_map, actual_bad_dist_list):
    generated_bad_dist_list = tiling_utils.find_manual_auto_invalid_dist(manual_to_auto_map)

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
        test_cases._ANNOT_SAMPLE_MAPPING, check_dist, check_duplicates, check_mismatches
    )

    actual_annot = test_cases.generate_sample_annot(check_dist, check_duplicates, check_mismatches)

    assert generated_annot == actual_annot


@parametrize('randomize_setting', [False, True])
@parametrize('moly_insert, moly_interval', test_cases._REMAP_MOLY_INTERVAL_CASES)
@parametrize('moly_path', [param('bad_moly_point.json', marks=file_missing_err),
                           param('sample_moly_point.json')])
def test_remap_and_reorder_fovs(moly_path, randomize_setting, moly_insert, moly_interval):
    # define the sample Moly point
    sample_moly_point = test_utils.generate_sample_fov_tiling_entry(
        coord=(14540, -10830), name="MoQC"
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
            manual_coords, manual_fov_names
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
        # NOTE: the randomization test fails once in a blue moon due to how randomization works
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
