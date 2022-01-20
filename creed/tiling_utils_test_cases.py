import numpy as np

from creed.tiling_utils import XYCoord

# for tiled region param dict creation
_PARAM_SET_MOLY_INTERVAL_VALUE_CASES = [0, 1]


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


# testing failures for TMA fov generation
class TMAFovListFailureCases:
    def case_json_path_failure(self):
        return 'bad_json.path', 3, 3

    def case_x_fov_failure(self):
        return 'sample_tma_corners.json', 2, 3

    def case_y_fov_failure(self):
        return 'sample_tma_corners.json', 3, 2

    def case_four_fovs_failure(self):
        return 'sample_tma_corners.json', 3, 3


# parameters for remapping
class RemapFOVOrderRandomizeCases:
    def case_false(self):
        return False

    def case_true(self):
        return True


class RemapMolyInsertCases:
    def case_false(self):
        return False

    def case_true(self):
        return True


class RemapMolyIntervalCases:
    def case_uneven_partition(self):
        return 4

    def case_even_partition(self):
        return 2


# testing failures for remapping
class RemappingFailureCases:
    def case_bad_moly_path(self):
        return 'bad_path.json', 2

    def case_bad_moly_interval(self):
        return 'sample_moly_point.json', 0
