from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from toffy.tiling_utils import XYCoord

param = pytest.param
parametrize = pytest.mark.parametrize
xfail = pytest.mark.xfail

# shortcuts to make the marks arg in pytest.params easier
value_err = [xfail(raises=ValueError, strict=True)]


def generate_fiducial_read_vals(user_input_type='none'):
    user_inputs = [1.5 * (i + 1) if i % 2 == 0 else 2 * i for i in np.arange(24)]

    if user_input_type == 'same_types':
        bad_inputs_to_insert = [-p for p in user_inputs if not isinstance(p, str)]
        for i in np.arange(0, len(user_inputs), 2):
            user_inputs.insert(int(i), bad_inputs_to_insert[int(i / 2)])

    if user_input_type == 'diff_types':
        bad_inputs_to_insert = [str(p) + '_bad' for p in user_inputs]
        for i in np.arange(0, len(user_inputs), 2):
            user_inputs.insert(int(i), bad_inputs_to_insert[int(i / 2)])

    return user_inputs


class FiducialInfoReadCases:
    def case_no_reentry(self):
        return generate_fiducial_read_vals()

    def case_reentry_same_type(self):
        return generate_fiducial_read_vals(user_input_type='same_types')

    def case_reentry_different_type(self):
        return generate_fiducial_read_vals(user_input_type='diff_types')


# define the list of region start coords and names
_TILED_REGION_ROI_COORDS = [(50, 150), (100, 300)]
_TILED_REGION_ROI_NAMES = ["TheFirstROI", "TheSecondROI"]
_TILED_REGION_ROI_SIZES = [1000, 2000]


# this function assumes that ROI 2's corresponding values are linearly spaced from ROI 1's
# NOTE: x and y correspond to column and row index respectively as specified in the JSON spec file
def generate_tiled_region_params(start_x_roi_1=50, start_y_roi_1=150,
                                 num_row_roi_1=4, num_col_roi_1=2,
                                 row_size_roi_1=2, col_size_roi_1=1, num_rois=2,
                                 roi_names=deepcopy(_TILED_REGION_ROI_NAMES)):
    # define this dictionary for testing purposes to ensure that function calls
    # equal what would be placed in param_set_values
    base_param_values = {
        'region_start_row': start_y_roi_1,
        'region_start_col': start_x_roi_1,
        'fov_num_row': num_row_roi_1,
        'fov_num_col': num_col_roi_1,
        'row_fov_size': row_size_roi_1,
        'col_fov_size': col_size_roi_1
    }

    # define the values for each param that should be contained for each ROI
    full_param_set = {
        param: list(np.arange(
            base_param_values[param],
            base_param_values[param] * (num_rois + 1),
            base_param_values[param]
        ))

        for param in base_param_values
    }

    # set the names for each ROI
    full_param_set['region_name'] = roi_names

    # TODO: might want to return just one and have the test function generate the other
    return base_param_values, full_param_set


# test tiled region parameter setting and FOV generation
# a helper function for generating params specific to each ROI for TiledRegionReadCases
# NOTE: the param moly_roi applies across all ROIs, so it's not set here
def generate_tiled_region_cases(roi_coord_list, roi_name_list, roi_sizes,
                                user_input_type='none', num_row_roi_1=4, num_col_roi_1=2,
                                random_roi_1='n', random_roi_2='Y'):
    # define the base value for each parameter to use for testing
    # as well as the full set of parameters for each FOV
    base_param_values, full_param_set = generate_tiled_region_params(
        roi_coord_list[0][0], roi_coord_list[0][1], num_row_roi_1, num_col_roi_1,
        roi_sizes[0], roi_sizes[0], len(roi_coord_list)
    )

    full_param_set['region_rand'] = ['N', 'Y']

    # define the list of user inputs to pass into the input functions for tiled regions
    user_inputs = [
        num_row_roi_1, num_col_roi_1, random_roi_1,
        num_row_roi_1 * 2, num_col_roi_1 * 2, random_roi_2
    ]

    # insert some bad inputs for the desire test type
    # want to test both invalid value inputs and invalid type inputs
    if user_input_type == 'same_types':
        bad_inputs_to_insert = [-1, 0, 'o', -1, 0, 'hello']
        for i in np.arange(0, len(user_inputs), 2):
            user_inputs.insert(int(i), bad_inputs_to_insert[int(i / 2)])

    elif user_input_type == 'diff_types':
        bad_inputs_to_insert = ['hello', 0, 5, -1, 'hello', 2.5]
        for i in np.arange(0, len(user_inputs), 2):
            user_inputs.insert(int(i), bad_inputs_to_insert[int(i / 2)])

    return roi_coord_list, roi_name_list, roi_sizes, user_inputs, base_param_values, full_param_set


# NOTE: because of the way the moly_interval param is handled
# it is generated directly in the tiling_utils_test test function
class TiledRegionReadCases:
    def case_no_reentry_no_moly_param(self):
        return generate_tiled_region_cases(
            _TILED_REGION_ROI_COORDS, _TILED_REGION_ROI_NAMES, deepcopy(_TILED_REGION_ROI_SIZES)
        )

    def case_no_reentry_with_moly_param(self):
        fcl, fnl, fs, ui, bpv, fps = generate_tiled_region_cases(
            _TILED_REGION_ROI_COORDS, _TILED_REGION_ROI_NAMES, deepcopy(_TILED_REGION_ROI_SIZES)
        )

        return fcl, fnl, fs, ui + ['Y'], bpv, fps

    def case_reentry_same_type_no_moly_param(self):
        return generate_tiled_region_cases(
            _TILED_REGION_ROI_COORDS, _TILED_REGION_ROI_NAMES, deepcopy(_TILED_REGION_ROI_SIZES),
            user_input_type='same_types'
        )

    def case_reentry_same_type_with_moly_param(self):
        fcl, fnl, fs, ui, bpv, fps = generate_tiled_region_cases(
            _TILED_REGION_ROI_COORDS, _TILED_REGION_ROI_NAMES, deepcopy(_TILED_REGION_ROI_SIZES),
            user_input_type='same_types'
        )

        return fcl, fnl, fs, ui + ['hello', 'Y'], bpv, fps

    def case_reentry_different_type_no_moly_param(self):
        return generate_tiled_region_cases(
            _TILED_REGION_ROI_COORDS, _TILED_REGION_ROI_NAMES, deepcopy(_TILED_REGION_ROI_SIZES),
            user_input_type='diff_types'
        )

    def case_reentry_different_type_with_moly_param(self):
        fcl, fnl, fs, ui, bpv, fps = generate_tiled_region_cases(
            _TILED_REGION_ROI_COORDS, _TILED_REGION_ROI_NAMES, deepcopy(_TILED_REGION_ROI_SIZES),
            user_input_type='diff_types'
        )

        return fcl, fnl, fs, ui + [-2.5, 'Y'], bpv, fps

    @xfail(raises=ValueError, strict=True)
    def case_bad_fov_size_value_no_moly_param(self):
        fcl, fnl, fs, ui, bpv, fps = generate_tiled_region_cases(
            _TILED_REGION_ROI_COORDS, _TILED_REGION_ROI_NAMES, deepcopy(_TILED_REGION_ROI_SIZES)
        )

        fs[0] = -5

        return fcl, fnl, fs, ui, bpv, fps

    @xfail(raises=ValueError, strict=True)
    def case_bad_fov_size_value_moly_param(self):
        fcl, fnl, fs, ui, bpv, fps = generate_tiled_region_cases(
            _TILED_REGION_ROI_COORDS, _TILED_REGION_ROI_NAMES, deepcopy(_TILED_REGION_ROI_SIZES)
        )

        fs[0] = -5

        return fcl, fnl, fs, ui + ['Y'], bpv, fps


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
# TODO: make one mapping dict for all of the tests below?
class MappingDistanceCases:
    def case_no_bad_dist(self):
        manual_to_auto_map = {
            'R0C0': 'R0C0',
            'R0C1': 'R0C1',
            'R0C2': 'R0C1',
            'R1C0': 'R1C0',
            'R1C1': 'R1C2',
            'R1C2': 'R1C2'
        }

        manual_auto_dist = pd.DataFrame(
            np.vstack([
                [5.0, 0, 0, 0, 0, 0],
                [0, 2.0, 0, 0, 0, 0],
                [0, 49.0, 0, 0, 0, 0],
                [0, 0, 0, 50.0, 0, 0],   # NOTE: error only if STRICTLY GREATER than dist_threshold
                [0, 0, 0, 0, 0, 49.99],
                [0, 0, 0, 0, 0, 35.0]
            ]),
            index=list(manual_to_auto_map.keys()),
            columns=['R0C0', 'R0C1', 'R0C2', 'R1C0', 'R1C1', 'R1C2']
        )

        bad_dist_list = []

        return manual_to_auto_map, manual_auto_dist, bad_dist_list

    def case_bad_dist(self):
        manual_to_auto_map = {
            'R0C0': 'R0C0',
            'R0C1': 'R0C1',
            'R0C2': 'R0C1',
            'R1C0': 'R1C0',
            'R1C1': 'R1C2',
            'R1C2': 'R1C2'
        }

        manual_auto_dist = pd.DataFrame(
            np.vstack([
                [5.0, 0, 0, 0, 0, 0],
                [0, 55.7, 0, 0, 0, 0],
                [0, 49.0, 0, 0, 0, 0],
                [0, 0, 0, 50.1, 0, 0],
                [0, 0, 0, 0, 0, 75.0],
                [0, 0, 0, 0, 0, 35.0]
            ]),
            index=list(manual_to_auto_map.keys()),
            columns=['R0C0', 'R0C1', 'R0C2', 'R1C0', 'R1C1', 'R1C2']
        )

        # this list will always be sorted by distance descending
        bad_dist_list = [('R1C1', 'R1C2', 75.00), ('R0C1', 'R0C1', 55.70), ('R1C0', 'R1C0', 50.10)]

        return manual_to_auto_map, manual_auto_dist, bad_dist_list


# testing auto FOVs mapped to by multiple manual FOVs validation
class MappingDuplicateCases:
    def case_no_duplicates(self):
        manual_to_auto_map = {
            'R0C0': 'R0C0',
            'R0C1': 'R0C1',
            'R0C2': 'R0C3',
            'R1C0': 'R1C0',
            'R1C1': 'R1C2',
            'R1C2': 'R1C4'
        }

        duplicate_list = []

        return manual_to_auto_map, duplicate_list

    def case_at_most_one_duplicate(self):
        manual_to_auto_map = {
            'R0C0': 'R0C0',
            'R0C1': 'R0C1',
            'R0C2': 'R0C3',
            'R1C0': 'R0C1',
            'R1C1': 'R1C2',
            'R1C2': 'R1C2'
        }

        duplicate_list = [
            ('R0C1', ('R0C1', 'R1C0')), ('R1C2', ('R1C1', 'R1C2'))
        ]

        return manual_to_auto_map, duplicate_list

    def case_more_than_one_duplicate(self):
        manual_to_auto_map = {
            'R0C0': 'R0C1',
            'R0C1': 'R0C1',
            'R0C2': 'R0C3',
            'R1C0': 'R0C1',
            'R1C1': 'R1C2',
            'R1C2': 'R1C2'
        }

        duplicate_list = [
            ('R0C1', ('R0C0', 'R0C1', 'R1C0')), ('R1C2', ('R1C1', 'R1C2'))
        ]

        return manual_to_auto_map, duplicate_list


# testing manual-auto FOV name mismatch validation
class MappingMismatchCases:
    def case_no_mismatches(self):
        manual_to_auto_map = {
            'R0C0': 'R0C0',
            'R0C1': 'R0C1',
            'R1C0': 'R1C0',
            'R1C1': 'R1C1'
        }

        mismatch_list = []

        return manual_to_auto_map, mismatch_list

    def case_mismatches(self):
        manual_to_auto_map = {
            'R0C0': 'R0C0',
            'R0C1': 'R0C2',
            'R1C0': 'R1C1',
            'R1C1': 'R1C1'
        }

        mismatch_list = [('R0C1', 'R0C2'), ('R1C0', 'R1C1')]

        return manual_to_auto_map, mismatch_list


_ANNOT_SAMPLE_MAPPING = {
    'R0C0': 'R0C0',
    'R0C1': 'R0C1',
    'R0C2': 'R1C0',
    'R1C0': 'R1C0',
    'R1C1': 'R0C2',
    'R1C2': 'R0C2'
}

_ANNOT_SAMPLE_DIST = pd.DataFrame(
    np.vstack([
        [1.5, 0, 0, 0, 0, 0],  # no errors
        [0, 60.0, 0, 0, 0, 0],   # just a distance error
        [0, 0, 0, 49.0, 0, 0],   # just a name mismatch error
        [0, 0, 0, 5.0, 0, 0],   # since it's the 2nd FOV mapped to R1C0, throws a duplicate error
        [0, 0, 100.0, 0, 0, 0],  # distance error and name mismatch error
        [0, 0, 87.5, 0, 0, 0]   # generates all 3 errors
    ]),
    index=list(_ANNOT_SAMPLE_MAPPING.keys()),
    columns=['R0C0', 'R0C1', 'R0C2', 'R1C0', 'R1C1', 'R1C2']
)


# a helper function to generate the sample annotation needed for _ANNOT_SAMPLE_MAPPING
# NOTE: based on distance values in _ANNOT_SAMPLE_DIST
def generate_sample_annot(check_dist, check_duplicates, check_mismatches):
    annot = ""

    if check_dist is not None:
        annot += \
            "The following mappings are placed more than %d microns apart:\n\n" % check_dist
        annot += "User-defined FOV R1C1 to TMA-grid FOV R0C2 (distance: 100.00)\n"
        annot += "User-defined FOV R1C2 to TMA-grid FOV R0C2 (distance: 87.50)\n"
        annot += "User-defined FOV R0C1 to TMA-grid FOV R0C1 (distance: 60.00)"
        annot += "\n\n"

    if check_duplicates:
        annot += \
            "The following TMA-grid FOVs have more than one user-defined FOV mapping to it:\n\n"
        annot += "TMA-grid FOV R0C2: mapped with user-defined FOVs R1C1, R1C2\n"
        annot += "TMA-grid FOV R1C0: mapped with user-defined FOVs R0C2, R1C0"
        annot += "\n\n"

    if check_mismatches:
        annot += \
            "The following mappings have mismatched names:\n\n"
        annot += "User-defined FOV R0C2: mapped with TMA-grid FOV R1C0\n"
        annot += "User-defined FOV R1C1: mapped with TMA-grid FOV R0C2\n"
        annot += "User-defined FOV R1C2: mapped with TMA-grid FOV R0C2"
        annot += "\n\n"

    return annot


# testing Moly point insertion for remapping, this one's long so don't put directly in decorator
_REMAP_MOLY_INTERVAL_CASES = [
    param(True, 2.5, marks=value_err),
    param(True, 0, marks=value_err),
    param(False, 4),
    param(True, 4),
    param(False, 2),
    param(True, 2)
]
