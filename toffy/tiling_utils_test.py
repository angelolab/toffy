import copy
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

parametrize = pytest.mark.parametrize


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

    # tests a few things
    # 1. valid metadata keys are copied over from input_dict to output_dict
    # 2. keys_ignore do not make it into output_dict
    # 3. if a metadata key in input_dict exists in output_dict, it gets overwritten
    # 4. everything in output_dict that shouldn't get overwritten stays the same
    # 5. do not copy over non str, bool, int, or float values (ex. dict)
    # 6. if a value in keys_ignore doesn't exist in input_dict, ignore
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
    # test 1: int inputs
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

    # test 2: str inputs
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


def test_read_tiled_region_inputs(monkeypatch):
    # define a sample fovs list
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=[(0, 150), (100, 300)], fov_names=["TheFirstFOV", "TheSecondFOV"]
    )

    # define sample region_params to read data into
    sample_region_params = {rpf: [] for rpf in settings.REGION_PARAM_FIELDS}

    # intentionally place some lowercase letters as this function should support those
    user_inputs = iter([2, 4, 1, 2, 'N', 4, 8, 2, 4, 'y'])

    # override the default functionality of the input function
    monkeypatch.setattr('builtins.input', lambda _: next(user_inputs))

    # use the dummy user data to read values into the params lists
    tiling_utils.read_tiled_region_inputs(
        sample_fovs_list, sample_region_params
    )

    assert (sample_region_params[i]['region_start_x'] == 100 * i
            for i in range(len(sample_region_params)))
    assert (sample_region_params[i]['region_start_y'] == 150 * (i + 1)
            for i in range(len(sample_region_params)))
    assert (sample_region_params[i]['fov_num_x'] == 2 * (i + 1)
            for i in range(len(sample_region_params)))
    assert (sample_region_params[i]['fov_num_y'] == 4 * (i + 1)
            for i in range(len(sample_region_params)))
    assert (sample_region_params[i]['x_fov_size'] == 1 * (i + 1)
            for i in range(len(sample_region_params)))
    assert (sample_region_params[i]['y_fov_size'] == 2 * (i + 1)
            for i in range(len(sample_region_params)))

    assert sample_region_params['region_rand'] == ['N', 'Y']


def test_generate_region_info():
    sample_region_inputs = {
        'region_start_x': [100, 200],
        'region_start_y': [200, 400],
        'fov_num_x': [3, 6],
        'fov_num_y': [6, 12],
        'x_fov_size': [5, 10],
        'y_fov_size': [10, 20],
        'region_rand': ['N', 'Y']
    }

    # generate the region params
    sample_region_params = tiling_utils.generate_region_info(sample_region_inputs)

    # assert region_start_x set correctly
    assert all(
        sample_region_params[i]['region_start_x'] == 100 * (i + 1)
        for i in range(len(sample_region_params))
    )

    # assert region_start_y set correctly
    assert all(
        sample_region_params[i]['region_start_y'] == 200 * (i + 1)
        for i in range(len(sample_region_params))
    )

    # assert fov_num_x set correctly
    assert all(
        sample_region_params[i]['fov_num_x'] == 3 * (i + 1)
        for i in range(len(sample_region_params))
    )

    # assert fov_num_y set correctly
    assert all(
        sample_region_params[i]['fov_num_y'] == 6 * (i + 1)
        for i in range(len(sample_region_params))
    )

    # assert x_fov_size set correctly
    assert all(
        sample_region_params[i]['x_fov_size'] == 5 * (i + 1)
        for i in range(len(sample_region_params))
    )

    # assert y_fov_size set correctly
    assert all(
        sample_region_params[i]['y_fov_size'] == 10 * (i + 1)
        for i in range(len(sample_region_params))
    )

    # assert region_rand set correctly
    assert sample_region_params[0]['region_rand'] == 'N'
    assert sample_region_params[1]['region_rand'] == 'Y'


@parametrize('moly_interval_val', [0, 1])
def test_set_tiled_region_params(monkeypatch, moly_interval_val):
    # bad fov list path provided
    with pytest.raises(FileNotFoundError):
        tiling_utils.set_tiled_region_params('bad_fov_list_path.json')

    # define a sample set of fovs
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=[(0, 150), (100, 300)], fov_names=["TheFirstFOV", "TheSecondFOV"]
    )

    # set the user inputs
    # intentionally place some lowercase letters as this function should support those
    user_inputs = iter([2, 4, 1, 2, 'n', 4, 8, 2, 4, 'Y', 'y', moly_interval_val])

    # override the default functionality of the input function
    monkeypatch.setattr('builtins.input', lambda _: next(user_inputs))

    with tempfile.TemporaryDirectory() as temp_dir:
        # write fov list
        sample_fov_list_path = os.path.join(temp_dir, 'fov_list.json')
        with open(sample_fov_list_path, 'w') as fl:
            json.dump(sample_fovs_list, fl)

        # run tiling parameter setting process with predefined user inputs
        sample_tiling_params = tiling_utils.set_tiled_region_params(sample_fov_list_path)

        # assert the fovs in the tiling params are the same as in the original fovs list
        assert sample_tiling_params['fovs'] == sample_fovs_list['fovs']

        # assert region start x and region start y values are correct
        sample_region_params = sample_tiling_params['region_params']
        fov_0 = sample_fovs_list['fovs'][0]
        fov_1 = sample_fovs_list['fovs'][1]

        # assert region start, fov_num, and fov_size set correctly
        assert (sample_region_params[i]['region_start_x'] == 100 * i
                for i in range(len(sample_region_params)))
        assert (sample_region_params[i]['region_start_y'] == 150 * (i + 1)
                for i in range(len(sample_region_params)))
        assert (sample_region_params[i]['fov_num_x'] == 2 * (i + 1)
                for i in range(len(sample_region_params)))
        assert (sample_region_params[i]['fov_num_y'] == 4 * (i + 1)
                for i in range(len(sample_region_params)))
        assert (sample_region_params[i]['x_fov_size'] == 1 * (i + 1)
                for i in range(len(sample_region_params)))
        assert (sample_region_params[i]['y_fov_size'] == 2 * (i + 1)
                for i in range(len(sample_region_params)))

        # assert region_rand set correctly
        assert sample_region_params[0]['region_rand'] == 'N'
        assert sample_region_params[1]['region_rand'] == 'Y'

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


# TODO: clean up this test, something like Adam's setup might work here
@parametrize_with_cases('coords, actual_pairs', cases=test_cases.RhombusCoordInputTests)
def test_generate_x_y_fov_pairs_rhombus(coords, actual_pairs):
    # retrieve the coordinates defining the TMA and the number of FOVs along each axis
    top_left, top_right, bottom_left, bottom_right = coords

    # generate the FOV-coordinate pairs
    pairs = tiling_utils.generate_x_y_fov_pairs_rhombus(
        top_left, top_right, bottom_left, bottom_right, 4, 3
    )

    assert pairs == actual_pairs


def test_generate_tiled_region_fov_list_failure():
    # moly point file path validation
    with pytest.raises(FileNotFoundError):
        tiling_utils.generate_tiled_region_fov_list({}, 'bad_moly_path.json')


@parametrize_with_cases('randomize_setting', cases=test_cases.TiledRegionRandomizeCases)
@parametrize_with_cases(
    'moly_region_setting, moly_interval_setting, moly_interval_value, '
    'moly_insert_indices, fov_1_end_pos', cases=test_cases.TiledRegionMolySettingCases
)
def test_generate_tiled_region_fov_list_success(randomize_setting, moly_region_setting,
                                                moly_interval_setting, moly_interval_value,
                                                moly_insert_indices, fov_1_end_pos):
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
            sample_tiling_params, sample_moly_path
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
                        cases=test_cases.ValidateRhombusCoordsTests, glob='*_failure')
def test_validate_tma_corners_failure(top_left, top_right, bottom_left, bottom_right):
    # error checks 1-4: test invalid top_left, top_right, bottom_left, bottom_right respectively
    with pytest.raises(ValueError):
        tiling_utils.validate_tma_corners(top_left, top_right, bottom_left, bottom_right)


@parametrize_with_cases('top_left, top_right, bottom_left, bottom_right',
                        cases=test_cases.ValidateRhombusCoordsTests, glob='*_success')
def test_validate_tma_corners_success(top_left, top_right, bottom_left, bottom_right):
    # this should not throw an error
    tiling_utils.validate_tma_corners(top_left, top_right, bottom_left, bottom_right)


@parametrize_with_cases('tma_corners_file, num_x, num_y, err_type, err_match',
                        cases=test_cases.TMAFovListFailureCases)
def test_generate_tma_fov_list_failure(tma_corners_file, num_x, num_y, err_type, err_match):
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=[(1500, 2000), (2000, 2000), (1500, 1000), (2000, 1000),
                    (100, 100), (200, 200)],
        fov_names=["TheFirstFOV"] * 4 + ["TheSecondFOV"] * 2
    )

    with tempfile.TemporaryDirectory() as td:
        sample_tma_corners_path = os.path.join(td, 'sample_tma_corners.json')

        with open(sample_tma_corners_path, 'w') as sfl:
            json.dump(sample_fovs_list, sfl)

        # test 1: tma_corners_file does not exist
        # test 2: invalid num_x
        # test 3: invalid num_y
        # test 4: the number of FOVs provided in tma_corners_file is not 4
        with pytest.raises(err_type, match=err_match):
            tiling_utils.generate_tma_fov_list(
                os.path.join(td, tma_corners_file), num_x, num_y
            )


@parametrize_with_cases('coords, actual_pairs', cases=test_cases.RhombusCoordInputTests)
def test_generate_tma_fov_list_success(coords, actual_pairs):
    # file path validation
    with pytest.raises(FileNotFoundError):
        tiling_utils.generate_tma_fov_list(
            'bad_path.json', 3, 3
        )

    # extract the coordinates
    top_left, top_right, bottom_left, bottom_right = coords

    # generate a sample FOVs list
    # NOTE: this intentionally contains more than 4 FOVs for now so it fails immediately
    # we will trim it later on
    sample_fovs_list = test_utils.generate_sample_fovs_list(
        fov_coords=[astuple(top_left), astuple(top_right),
                    astuple(bottom_left), astuple(bottom_right)],
        fov_names=["TheFirstFOV"] * 4
    )

    # save sample FOV
    with tempfile.TemporaryDirectory() as td:
        sample_tma_corners_path = os.path.join(td, 'sample_tma_corners.json')
        with open(sample_tma_corners_path, 'w') as sfl:
            json.dump(sample_fovs_list, sfl)

        # NOTE: we'll save the coordinate checks for test_validate_tma_corners

        # create the FOV regions
        num_x = 4
        num_y = 3
        fov_regions = tiling_utils.generate_tma_fov_list(
            sample_tma_corners_path, num_x, num_y
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
        # TODO: might be a duplicate test for corners above, might want this to handle it alone
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
    auto_coords = [(0, 0), (0, 50), (0, 100), (100, 0), (100, 50), (100, 100),
                   (150, 100), (150, 150)]
    auto_fov_names = ['row%d_col%d' % (x, y) for (x, y) in auto_coords]

    # generate the list of automatically-generated fovs
    auto_sample_fovs = dict(zip(auto_fov_names, auto_coords))

    # define the coordinates and fov names proposed by the user
    manual_coords = [(0, 25), (50, 25), (50, 50), (75, 50), (100, 25)]
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
        'row0_col25': 'row0_col0',
        'row50_col25': 'row0_col0',
        'row50_col50': 'row0_col100',
        'row75_col50': 'row100_col100',
        'row100_col25': 'row100_col0'
    }

    assert manual_to_auto_map == actual_map


def test_generate_fov_circles():
    # we'll be copying the data generated from test_assign_closest_fovs
    sample_manual_to_auto_map = {
        'row0_col25': 'row0_col0',
        'row50_col25': 'row0_col0',
        'row50_col50': 'row0_col50',
        'row75_col50': 'row100_col50',
        'row100_col25': 'row100_col0'
    }

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
        sample_manual_to_auto_map, sample_manual_fovs_info,
        sample_auto_fovs_info, 'row0_col25', 'row0_col0',
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


@parametrize_with_cases('moly_path, moly_interval, err_type, err_match',
                        cases=test_cases.RemappingFailureCases)
def test_remap_and_reorder_fovs_failure(moly_path, moly_interval, err_type, err_match):
    # define the sample Moly point
    sample_moly_point = test_utils.generate_sample_fov_tiling_entry(
        coord=(14540, -10830), name="MoQC"
    )

    # save the Moly point
    with open('sample_moly_point.json', 'w') as smp:
        json.dump(sample_moly_point, smp)

    # test 1: moly_path does not exist
    # test 2: moly_interval is less than 1
    with pytest.raises(err_type, match=err_match):
        tiling_utils.remap_and_reorder_fovs(
            {}, {}, moly_path, moly_insert=True, moly_interval=moly_interval)


@pytest.mark.parametrize('randomize_setting', test_cases._REMAP_FOV_ORDER_RANDOMIZE_CASES)
@pytest.mark.parametrize('moly_insert', test_cases._REMAP_MOLY_INSERT_CASES)
@pytest.mark.parametrize('moly_interval', test_cases._REMAP_MOLY_INTERVAL_CASES)
def test_remap_and_reorder_fovs_success(randomize_setting, moly_insert, moly_interval):
    # define the sample Moly point
    sample_moly_point = test_utils.generate_sample_fov_tiling_entry(
        coord=(14540, -10830), name="MoQC"
    )

    # save the Moly point
    with open('sample_moly_point.json', 'w') as smp:
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
        manual_sample_fovs_copy, sample_mapping, 'sample_moly_point.json', randomize_setting,
        moly_insert, moly_interval
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
