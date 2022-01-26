from copy import deepcopy
import numpy as np
import pytest

from toffy.tiling_utils import XYCoord

xfail = pytest.mark.xfail


# test tiled region parameter setting and FOV generation
# a helper function for generating params specific to each FOV for TiledRegionReadCases
# NOTE: the param moly_region applies across all FOVs, so it's not set here
def generate_tiled_region_cases(user_input_type='none'):
    # define the list of FOV starting coordinates and their corresponding names
    # NOTE: leaving these hard-coded for now, we can use additional params to adjust these
    fov_coord_list = [(50, 150), (100, 300)]
    fov_name_list = ["TheFirstFOV", "TheSecondFOV"]

    # define the numeric parameters for FOV 1, we can generate FOV 2's data
    # by simply multiplying FOV 1's data by 2
    num_x_fov_1 = 2
    num_y_fov_1 = 4
    x_size_fov_1 = 1
    y_size_fov_1 = 2

    num_x_fov_2 = num_x_fov_1 * 2
    num_y_fov_2 = num_y_fov_1 * 2
    x_size_fov_2 = x_size_fov_1 * 2
    y_size_fov_2 = y_size_fov_1 * 2

    # define the randomize parameters for FOV 1 and FOV 2
    # add some lowercase letters to ensure they work
    random_fov_1 = 'n'
    random_fov_2 = 'Y'

    # define the values for each param that should be contained for each FOV
    end_index = len(fov_coord_list) + 1
    param_set_values = {
        'region_start_x': [coord[0] for coord in fov_coord_list],
        'region_start_y': [coord[1] for coord in fov_coord_list],
        'fov_num_x': list(np.arange(
            num_x_fov_1, num_x_fov_1 * end_index, num_x_fov_1
        )),
        'fov_num_y': list(np.arange(
            num_y_fov_1, num_y_fov_1 * end_index, num_y_fov_1
        )),
        'x_fov_size': list(np.arange(
            x_size_fov_1, x_size_fov_1 * end_index, x_size_fov_1
        )),
        'y_fov_size': list(np.arange(
            y_size_fov_1, y_size_fov_1 * end_index, y_size_fov_1
        )),
        'region_rand': [random_fov_1.upper(), random_fov_2.upper()]
    }

    user_inputs = [
        num_x_fov_1, num_y_fov_1, x_size_fov_1, y_size_fov_1, random_fov_1,
        num_x_fov_2, num_y_fov_2, x_size_fov_2, y_size_fov_2, random_fov_2
    ]

    if user_input_type == 'same_type':
        bad_inputs_to_insert = [-1, 0, -2, -3, 'o', -1, 0, -2, -3, 'hello']
        for i in np.arange(0, len(user_inputs), 2):
            user_inputs.insert(int(i), bad_inputs_to_insert[int(i / 2)])

    elif user_input_type == 'diff_type':
        bad_inputs_to_insert = ['hello', 0, -2, 2.5, 5, -1, 'hello', 2.5, -3, 2.5]
        for i in np.arange(0, len(user_inputs), 2):
            user_inputs.insert(int(i), bad_inputs_to_insert[int(i / 2)])

    return fov_coord_list, fov_name_list, user_inputs, param_set_values


# NOTE: because of the way the moly_interval param is handled
# it is generated directly in the tiling_utils_test test function
class TiledRegionReadCases:
    def case_no_reentry_no_moly_param(self):
        return generate_tiled_region_cases()

    def case_no_reentry_with_moly_param(self):
        fcl, fnl, ui, psv = generate_tiled_region_cases()
        return fcl, fnl, ui + ['Y'], psv

    def case_reentry_same_type_no_moly_param(self):
        return generate_tiled_region_cases('same_type')

    def case_reentry_same_type_with_moly_param(self):
        fcl, fnl, ui, psv = generate_tiled_region_cases('same_type')
        return fcl, fnl, ui + ['hello', 'Y'], psv

    def case_reentry_different_type_no_moly_param(self):
        return generate_tiled_region_cases('diff_type')

    def case_reentry_different_type_with_moly_param(self):
        fcl, fnl, ui, psv = generate_tiled_region_cases('diff_type')
        return fcl, fnl, ui + [-2.5, 'Y'], psv


class TiledRegionMolySettingCases:
    @xfail(raises=FileNotFoundError, strict=True)
    def case_bad_moly_path(self):
        return 'bad_moly_point.json', 'N', False, 0, [], 8

    def case_no_region_no_interval(self):
        return 'sample_moly_point.json', 'N', False, 0, [], 8

    def case_no_region_interval_uneven_partition(self):
        return 'sample_moly_point.json', 'N', True, 3, [3, 7, 11, 15, 19], 10

    def case_no_region_interval_even_partition(self):
        return 'sample_moly_point.json', 'N', True, 4, [4, 13], 9

    def case_region_no_interval(self):
        return 'sample_moly_point.json', 'Y', False, 0, [8], 9

    def case_region_interval_uneven_partition(self):
        return 'sample_moly_point.json', 'Y', True, 3, [3, 7, 10, 12, 16, 20], 11

    def case_region_interval_even_partition(self):
        return 'sample_moly_point.json', 'Y', True, 4, [4, 9, 14], 10


# TMA rhombus coordinate validation
class ValidateRhombusCoordsTests:
    @xfail(raises=ValueError, strict=True)
    def case_top_left_failure(self):
        top_left = XYCoord(100, 200)
        top_right = XYCoord(50, 100)
        bottom_left = XYCoord(150, 300)
        bottom_right = XYCoord(100, 200)

        return top_left, top_right, bottom_left, bottom_right

    @xfail(raises=ValueError, strict=True)
    def case_bottom_left_failure(self):
        top_left = XYCoord(100, 200)
        top_right = XYCoord(150, 100)
        bottom_left = XYCoord(150, 300)
        bottom_right = XYCoord(100, 200)

        return top_left, top_right, bottom_left, bottom_right

    @xfail(raises=ValueError, strict=True)
    def case_top_right_failure(self):
        top_left = XYCoord(100, 200)
        top_right = XYCoord(150, 100)
        bottom_left = XYCoord(150, 300)
        bottom_right = XYCoord(200, 200)

        return top_left, top_right, bottom_left, bottom_right

    @xfail(raises=ValueError, strict=True)
    def case_bottom_right_failure(self):
        top_left = XYCoord(100, 400)
        top_right = XYCoord(150, 100)
        bottom_left = XYCoord(150, 300)
        bottom_right = XYCoord(200, 200)

        return top_left, top_right, bottom_left, bottom_right

    def case_success(self):
        top_left = XYCoord(100, 400)
        top_right = XYCoord(150, 300)
        bottom_left = XYCoord(150, 300)
        bottom_right = XYCoord(200, 200)

        return top_left, top_right, bottom_left, bottom_right


# TMA rhombus inputs for coordinate generation and TMA FOV list generation
_TMA_RHOMBUS_X_COORDS = (1500, 1666, 1833, 2000)


class RhombusCoordInputTests:
    def case_rectangle(self):
        coords = [
            XYCoord(1500, 2000),
            XYCoord(2000, 2000),
            XYCoord(1500, 1000),
            XYCoord(2000, 1000)
        ]

        # since this is a simple rectangle we can easily and programmatically define these coords
        actual_pairs = []
        for x_coord in _TMA_RHOMBUS_X_COORDS:
            actual_pairs.extend([
                (x_coord, y_coord)
                for y_coord in reversed(np.arange(1000, 2500, 500))
            ])

        return coords, actual_pairs

    def case_x_slant(self):
        coords = [
            XYCoord(1500, 2000),
            XYCoord(2000, 2000),
            XYCoord(1600, 1000),
            XYCoord(2200, 1000)
        ]

        # total x-offset is 150 (average of the left-offset (100) and right-offset (200))
        # 3 fovs along the y-axis, so divide by 2 to get individual fov offset
        x_increment = ((coords[2].x - coords[0].x) + (coords[3].x - coords[1].x)) / 2
        x_increment = x_increment / 2

        actual_pairs = []
        for x_coord in _TMA_RHOMBUS_X_COORDS:
            actual_pairs.extend([
                (x_coord + x_increment * i, y_coord)
                for (i, y_coord) in enumerate(reversed(np.arange(1000, 2500, 500)))
            ])

        return coords, actual_pairs

    def case_y_slant(self):
        coords = [
            XYCoord(1500, 2000),
            XYCoord(2000, 2225),
            XYCoord(1500, 1000),
            XYCoord(2000, 1100)
        ]

        # total y-offset is 162.5 (average of the top-offset (225) and right_offset (100))
        # 4 fovs along the x-axis, so divide by 3 to get individual fov offset
        y_increment = ((coords[1].y - coords[0].y) + (coords[3].y - coords[2].y)) / 2
        y_increment = y_increment / 3

        actual_pairs = []
        for i, x_coord in enumerate(_TMA_RHOMBUS_X_COORDS):
            actual_pairs.extend([
                (x_coord, int(y_coord + y_increment * i))
                for y_coord in reversed(np.arange(1000, 2500, 500))
            ])

        return coords, actual_pairs

    def case_both_slant(self):
        coords = [
            XYCoord(1600, 2000),
            XYCoord(2200, 2175),
            XYCoord(1475, 1000),
            XYCoord(2100, 1200)
        ]

        # total x-offset is -112.5 (average of the left-offset (-125) and right-offset (-100))
        # 3 fovs along the y-axis, so divide by 2 to get the individual fov offset
        x_increment = ((coords[2].x - coords[0].x) + (coords[3].x - coords[1].x)) / 2
        x_increment = x_increment / 2

        # total y-offset is 187.5 (average of the top-offset (175) and the bottom-offset (200))
        # 4 fovs along the x-axis, so divide by 3 to get the individual fov offset
        y_increment = ((coords[1].y - coords[0].y) + (coords[3].y - coords[2].y)) / 2
        y_increment = y_increment / 3

        # defining the equivalent rectangle indices will help for this
        # the baseline x-coord for a rectangle
        # take the difference between the x-value for the top-left and top-right corner (600)
        # since there are 4 fovs along the x-axis, divide by 3 (200)
        x_start = (coords[1].x - coords[0].x) / 3

        # the baseline y-coord for a rectangle
        # take the difference between the y-value for the bottom-left and top-left corner (-1000)
        # since there are 3 fovs along the y-axis, divide by 2 (-500)
        y_start = (coords[2].y - coords[0].y) / 2

        # start generating pairs from the top-left coordinate (1600, 2000)
        actual_pairs = [
            (int(coords[0].x + x_start * i + x_increment * j),
             int(coords[0].y + y_start * j + y_increment * i))
            for i in np.arange(4) for j in np.arange(3)
        ]

        return coords, actual_pairs


# testing failures for TMA fov generation
class TMAFovListFailureCases:
    def case_json_path_failure(self):
        return 'bad_json.path', 3, 3, FileNotFoundError, 'TMA corners file'

    def case_x_fov_failure(self):
        return 'sample_tma_corners.json', 2, 3, ValueError, 'x-axis'

    def case_y_fov_failure(self):
        return 'sample_tma_corners.json', 3, 2, ValueError, 'y-axis'

    def case_four_fovs_failure(self):
        return 'sample_tma_corners.json', 3, 3, ValueError, 'four FOVs'


# parameters for remapping
_REMAP_FOV_ORDER_RANDOMIZE_CASES = [False, True]
_REMAP_MOLY_INSERT_CASES = [False, True]
_REMAP_MOLY_INTERVAL_CASES = [4, 2]


# testing failures for remapping
class RemappingFailureCases:
    def case_bad_moly_path(self):
        return 'bad_path.json', 2, FileNotFoundError, 'Moly point'

    def case_bad_moly_interval_type(self):
        return 'sample_moly_point.json', 1.5, ValueError, 'moly_interval'

    def case_bad_moly_interval_value(self):
        return 'sample_moly_point.json', 0, ValueError, 'moly_interval'
