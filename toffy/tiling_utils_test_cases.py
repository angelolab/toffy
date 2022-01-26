import numpy as np

from toffy.tiling_utils import XYCoord


# for tiled region FOV generation
class TiledRegionRandomizeCases:
    def case_both_not_random(self):
        return ['N', 'N']

    def case_fov2_random(self):
        return ['N', 'Y']

    def case_both_random(self):
        return ['Y', 'Y']


class TiledRegionMolySettingCases:
    def case_no_region_no_interval(self):
        return 'N', False, 0, [], 8

    def case_no_region_interval_uneven_partition(self):
        return 'N', True, 3, [3, 7, 11, 15, 19], 10

    def case_no_region_interval_even_partition(self):
        return 'N', True, 4, [4, 13], 9

    def case_region_no_interval(self):
        return 'Y', False, 0, [8], 9

    def case_region_interval_uneven_partition(self):
        return 'Y', True, 3, [3, 7, 10, 12, 16, 20], 11

    def case_region_interval_even_partition(self):
        return 'Y', True, 4, [4, 9, 14], 10


# TMA rhombus coordinate validation
class ValidateRhombusCoordsTests:
    def case_top_left_failure(self):
        top_left = XYCoord(100, 200)
        top_right = XYCoord(50, 100)
        bottom_left = XYCoord(150, 300)
        bottom_right = XYCoord(100, 200)

        return top_left, top_right, bottom_left, bottom_right

    def case_bottom_left_failure(self):
        top_left = XYCoord(100, 200)
        top_right = XYCoord(150, 100)
        bottom_left = XYCoord(150, 300)
        bottom_right = XYCoord(100, 200)

        return top_left, top_right, bottom_left, bottom_right

    def case_top_right_failure(self):
        top_left = XYCoord(100, 200)
        top_right = XYCoord(150, 100)
        bottom_left = XYCoord(150, 300)
        bottom_right = XYCoord(200, 200)

        return top_left, top_right, bottom_left, bottom_right

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
        # 3 fovs along the y-axis, so divide by 2 to get individual fov offset (75)
        x_increment = 75
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
        y_increment = 162.5 / 3
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
        x_increment = -112.5 / 2

        # total y-offset is 187.5 (average of the top-offset (175) and the bottom-offset (200))
        # 4 fovs along the x-axis, so divide by 3 to get the individual fov offset
        y_increment = 187.5 / 3

        # defining the equivalent rectangle indices will help for this
        # the baseline x-coord for a rectangle
        # take the difference between the x-value for the top-left and top-right corner (600)
        # since there are 4 fovs along the x-axis, divide by 3 (200)
        x_start = 200

        # the baseline y-coord for a rectangle
        # take the difference between the y-value for the bottom-left and top-left corner (-1000)
        # since there are 3 fovs along the y-axis, divide by 2 (-500)
        y_start = -500

        # start generating pairs from the top-left coordinate (1600, 2000)
        actual_pairs = [
            (int(1600 + x_start * i + x_increment * j),
             int(2000 + y_start * j + y_increment * i))
            for i in np.arange(4) for j in np.arange(3)
        ]

        return coords, actual_pairs


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
