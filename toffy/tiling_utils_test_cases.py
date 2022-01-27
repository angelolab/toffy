from copy import deepcopy
import numpy as np
import pytest

from toffy.tiling_utils import XYCoord

param = pytest.param
parametrize = pytest.mark.parametrize
xfail = pytest.mark.xfail

# shortcuts to make the marks arg in pytest.params easier
value_err = [xfail(raises=ValueError, strict=True)]


# this function assumes that FOV 2's corresponding values are linearly spaced from
# TODO: do test functions need a docstring?
def generate_tiled_region_params(start_x_fov_1=50, start_y_fov_1=150, num_x_fov_1=2, num_y_fov_1=4,
                                 x_size_fov_1=1, y_size_fov_1=2, num_fovs=2):
    # define this dictionary for testing purposes to ensure that function calls
    # equal what would be placed in param_set_values
    base_param_values = {
        'region_start_x': start_x_fov_1,
        'region_start_y': start_y_fov_1,
        'fov_num_x': num_x_fov_1,
        'fov_num_y': num_y_fov_1,
        'x_fov_size': x_size_fov_1,
        'y_fov_size': y_size_fov_1
    }

    # define the values for each param that should be contained for each FOV
    full_param_set = {
        param: list(np.arange(
            base_param_values[param],
            base_param_values[param] * (num_fovs + 1),
            base_param_values[param]
        ))

        for param in base_param_values
    }

    # TODO: might want to return just one and have the test function generate the other
    return base_param_values, full_param_set


# test tiled region parameter setting and FOV generation
# a helper function for generating params specific to each FOV for TiledRegionReadCases
# NOTE: the param moly_region applies across all FOVs, so it's not set here
def generate_tiled_region_cases(fov_coord_list, fov_name_list, user_input_type='none',
                                num_x_fov_1=2, num_y_fov_1=4, x_size_fov_1=1,
                                y_size_fov_1=2, random_fov_1='n', random_fov_2='Y'):
    # define the base value for each parameter to use for testing
    # as well as the full set of parameters for each FOV
    base_param_values, full_param_set = generate_tiled_region_params(
        fov_coord_list[0][0], fov_coord_list[0][1], num_x_fov_1, num_y_fov_1,
        x_size_fov_1, y_size_fov_1, len(fov_coord_list)
    )

    full_param_set['region_rand'] = ['N', 'Y']

    # define the list of user inputs to pass into the input functions for tiled regions
    user_inputs = [
        num_x_fov_1, num_y_fov_1, x_size_fov_1, y_size_fov_1, random_fov_1,
        num_x_fov_1 * 2, num_y_fov_1 * 2, x_size_fov_1 * 2, y_size_fov_1 * 2, random_fov_2
    ]

    # insert some bad inputs for the desire test type
    # want to test both invalid value inputs and invalid type inputs
    if user_input_type == 'same_types':
        bad_inputs_to_insert = [-1, 0, -2, -3, 'o', -1, 0, -2, -3, 'hello']
        for i in np.arange(0, len(user_inputs), 2):
            user_inputs.insert(int(i), bad_inputs_to_insert[int(i / 2)])

    elif user_input_type == 'diff_types':
        bad_inputs_to_insert = ['hello', 0, -2, 2.5, 5, -1, 'hello', 2.5, -3, 2.5]
        for i in np.arange(0, len(user_inputs), 2):
            user_inputs.insert(int(i), bad_inputs_to_insert[int(i / 2)])

    return fov_coord_list, fov_name_list, user_inputs, base_param_values, full_param_set


# define the list of region start coords and names
_TILED_REGION_FOV_COORDS = [(50, 150), (100, 300)]
_TILED_REGION_FOV_NAMES = ["TheFirstFOV", "TheSecondFOV"]


# NOTE: because of the way the moly_interval param is handled
# it is generated directly in the tiling_utils_test test function
class TiledRegionReadCases:
    def case_no_reentry_no_moly_param(self):
        return generate_tiled_region_cases(
            _TILED_REGION_FOV_COORDS, _TILED_REGION_FOV_NAMES
        )

    def case_no_reentry_with_moly_param(self):
        fcl, fnl, ui, bpv, fps = generate_tiled_region_cases(
            _TILED_REGION_FOV_COORDS, _TILED_REGION_FOV_NAMES
        )

        return fcl, fnl, ui + ['Y'], bpv, fps

    def case_reentry_same_type_no_moly_param(self):
        return generate_tiled_region_cases(
            _TILED_REGION_FOV_COORDS, _TILED_REGION_FOV_NAMES, user_input_type='same_types'
        )

    def case_reentry_same_type_with_moly_param(self):
        fcl, fnl, ui, bpv, fps = generate_tiled_region_cases(
            _TILED_REGION_FOV_COORDS, _TILED_REGION_FOV_NAMES, user_input_type='same_types'
        )

        return fcl, fnl, ui + ['hello', 'Y'], bpv, fps

    def case_reentry_different_type_no_moly_param(self):
        return generate_tiled_region_cases(
            _TILED_REGION_FOV_COORDS, _TILED_REGION_FOV_NAMES, user_input_type='diff_types'
        )

    def case_reentry_different_type_with_moly_param(self):
        fcl, fnl, ui, bpv, fps = generate_tiled_region_cases(
            _TILED_REGION_FOV_COORDS, _TILED_REGION_FOV_NAMES, user_input_type='diff_types'
        )

        return fcl, fnl, ui + [-2.5, 'Y'], bpv, fps


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
class ValidateRhombusCoordsCases:
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


class RhombusCoordInputCases:
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


# testing manual and auto FOVs too far apart validation
class MappingDistanceCases:
    def case_no_bad_dist(self):
        manual_to_auto_map = {
            'R0C0': {
                'closest_auto_fov': 'R0C0',
                'distance': 5.0
            },
            'R0C1': {
                'closest_auto_fov': 'R0C1',
                'distance': 2.0
            },
            'R0C2': {
                'closest_auto_fov': 'R0C3',
                'distance': 49.0
            },
            'R1C0': {
                'closest_auto_fov': 'R1C0',
                'distance': 50.0  # NOTE: error only thrown if STRICTLY GREATER than dist_threshold
            },
            'R1C1': {
                'closest_auto_fov': 'R1C2',
                'distance': 49.99
            },
            'R1C2': {
                'closest_auto_fov': 'R1C4',
                'distance': 35.0
            }
        }

        bad_dist_list = []

        return manual_to_auto_map, bad_dist_list

    def case_bad_dist(self):
        manual_to_auto_map = {
            'R0C0': {
                'closest_auto_fov': 'R0C0',
                'distance': 5.0
            },
            'R0C1': {
                'closest_auto_fov': 'R0C1',
                'distance': 55.7
            },
            'R0C2': {
                'closest_auto_fov': 'R0C3',
                'distance': 49.0
            },
            'R1C0': {
                'closest_auto_fov': 'R1C0',
                'distance': 50.1
            },
            'R1C1': {
                'closest_auto_fov': 'R1C2',
                'distance': 75.0
            },
            'R1C2': {
                'closest_auto_fov': 'R1C4',
                'distance': 35.0
            }
        }

        # this list will always be sorted by distance descending
        bad_dist_list = [('R1C1', 'R1C2', 75.00), ('R0C1', 'R0C1', 55.70), ('R1C0', 'R1C0', 50.10)]

        return manual_to_auto_map, bad_dist_list


# testing auto FOVs mapped to by multiple manual FOVs validation
class MappingDuplicateCases:
    def case_no_duplicates(self):
        manual_to_auto_map = {
            'R0C0': {
                'closest_auto_fov': 'R0C0',
                'distance': 1.5
            },
            'R0C1': {
                'closest_auto_fov': 'R0C1',
                'distance': 3.0
            },
            'R0C2': {
                'closest_auto_fov': 'R0C3',
                'distance': 3.0
            },
            'R1C0': {
                'closest_auto_fov': 'R1C0',
                'distance': 1.0
            },
            'R1C1': {
                'closest_auto_fov': 'R1C2',
                'distance': 1.0
            },
            'R1C2': {
                'closest_auto_fov': 'R1C4',
                'distance': 1.0
            }
        }

        duplicate_list = []

        return manual_to_auto_map, duplicate_list

    def case_at_most_one_duplicate(self):
        manual_to_auto_map = {
            'R0C0': {
                'closest_auto_fov': 'R0C0',
                'distance': 1.5
            },
            'R0C1': {
                'closest_auto_fov': 'R0C1',
                'distance': 3.0
            },
            'R0C2': {
                'closest_auto_fov': 'R0C3',
                'distance': 3.0
            },
            'R1C0': {
                'closest_auto_fov': 'R0C1',
                'distance': 1.0
            },
            'R1C1': {
                'closest_auto_fov': 'R1C2',
                'distance': 1.0
            },
            'R1C2': {
                'closest_auto_fov': 'R1C2',
                'distance': 1.0
            }
        }

        duplicate_list = [
            ('R0C1', ('R0C1', 'R1C0')), ('R1C2', ('R1C1', 'R1C2'))
        ]

        return manual_to_auto_map, duplicate_list

    def case_more_than_one_duplicate(self):
        manual_to_auto_map = {
            'R0C0': {
                'closest_auto_fov': 'R0C1',
                'distance': 1.5
            },
            'R0C1': {
                'closest_auto_fov': 'R0C1',
                'distance': 3.0
            },
            'R0C2': {
                'closest_auto_fov': 'R0C3',
                'distance': 3.0
            },
            'R1C0': {
                'closest_auto_fov': 'R0C1',
                'distance': 1.0
            },
            'R1C1': {
                'closest_auto_fov': 'R1C2',
                'distance': 1.0
            },
            'R1C2': {
                'closest_auto_fov': 'R1C2',
                'distance': 1.0
            }
        }

        duplicate_list = [
            ('R0C1', ('R0C0', 'R0C1', 'R1C0')), ('R1C2', ('R1C1', 'R1C2'))
        ]

        return manual_to_auto_map, duplicate_list


# testing manual-auto FOV name mismatch validation
class MappingMismatchCases:
    def case_no_mismatches(self):
        manual_to_auto_map = {
            'R0C0': {
                'closest_auto_fov': 'R0C0',
                'distance': 1.5
            },
            'R0C1': {
                'closest_auto_fov': 'R0C1',
                'distance': 3.0
            },
            'R1C0': {
                'closest_auto_fov': 'R1C0',
                'distance': 1.0
            },
            'R1C1': {
                'closest_auto_fov': 'R1C1',
                'distance': 1.0
            }
        }

        mismatch_list = []

        return manual_to_auto_map, mismatch_list

    def case_mismatches(self):
        manual_to_auto_map = {
            'R0C0': {
                'closest_auto_fov': 'R0C0',
                'distance': 1.5
            },
            'R0C1': {
                'closest_auto_fov': 'R0C2',
                'distance': 3.0
            },
            'R1C0': {
                'closest_auto_fov': 'R1C1',
                'distance': 1.0
            },
            'R1C1': {
                'closest_auto_fov': 'R1C1',
                'distance': 1.0
            }
        }

        mismatch_list = [('R0C1', 'R0C2'), ('R1C0', 'R1C1')]

        return manual_to_auto_map, mismatch_list


# for the annotation test, define one manual-auto mapping
# TODO: make one dict for all the successes above?
_ANNOT_SAMPLE_MAPPING = manual_to_auto_map = {
    'R0C0': {  # no errors for both manual R0C0 and auto R0C0
        'closest_auto_fov': 'R0C0',
        'distance': 1.5
    },
    'R0C1': {   # just a distance error
        'closest_auto_fov': 'R0C1',
        'distance': 60.0
    },
    'R0C2': {   # just a mismatch error
        'closest_auto_fov': 'R1C0',
        'distance': 49.0
    },
    'R1C0': {   # no mismatch or distance errors, but auto R1C0 has two manual FOVs mapping to it
        'closest_auto_fov': 'R1C0',
        'distance': 5.0
    },
    'R1C1': {   # distance and mismatch error
        'closest_auto_fov': 'R2C0',
        'distance': 100.0
    },
    'R1C2': {   # generates all 3 errors
        'closest_auto_fov': 'R2C0',
        'distance': 87.5
    }
}


# a helper function to generate the sample annotation needed for _ANNOT_SAMPLE_MAPPING
# NOTE: hard code so as not to duplicate logic in generate_validation_annot
def generate_sample_annot(check_dist, check_duplicates, check_mismatches):
    annot = ""

    if check_dist is not None:
        annot += \
            "The following mappings are placed more than %d pixels apart:\n\n" % check_dist
        annot += "User-defined FOV R1C1 to TMA-grid FOV R2C0 (distance: 100.00)\n"
        annot += "User-defined FOV R1C2 to TMA-grid FOV R2C0 (distance: 87.50)\n"
        annot += "User-defined FOV R0C1 to TMA-grid FOV R0C1 (distance: 60.00)"
        annot += "\n\n"

    if check_duplicates:
        annot += \
            "The following TMA-grid FOVs have more than one user-defined FOV mapping to it:\n\n"
        annot += "TMA-grid FOV R1C0: mapped with user-defined FOVs R0C2, R1C0\n"
        annot += "TMA-grid FOV R2C0: mapped with user-defined FOVs R1C1, R1C2"
        annot += "\n\n"

    if check_mismatches:
        annot += \
            "The following mappings have mismatched names:\n\n"
        annot += "User-defined FOV R0C2: mapped with TMA-grid FOV R1C0\n"
        annot += "User-defined FOV R1C1: mapped with TMA-grid FOV R2C0\n"
        annot += "User-defined FOV R1C2: mapped with TMA-grid FOV R2C0"
        annot += "\n\n"

    return annot


# testing Moly point insertion for remapping
# NOTE: easier than enumerating these all in a class for moly interval verification
# this one's long so better here than directly in decorator
_REMAP_MOLY_INTERVAL_CASES = [
    param(True, 2.5, marks=value_err),
    param(True, 0, marks=value_err),
    param(False, 4),
    param(True, 4),
    param(False, 2),
    param(True, 2)
]
