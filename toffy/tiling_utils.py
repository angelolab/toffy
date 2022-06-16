import copy
from datetime import datetime
from IPython.display import display
import ipywidgets as widgets
from itertools import combinations, product
import json
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import os
import pandas as pd
import re
from typing import Optional
from skimage.draw import ellipse
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import warnings

from dataclasses import dataclass

from toffy import settings, json_utils
from ark.utils import misc_utils


def assign_metadata_vals(input_dict, output_dict, keys_ignore):
    """Copy the `str`, `int`, `float`, and `bool` metadata keys of
    `input_dict` over to `output_dict`, ignoring `keys_ignore` metadata keys

    Args:
        input_dict (dict):
            The `dict` to copy the metadata values over from
        output_dict (dict):
            The `dict` to copy the metadata values into.
            Note that if a metadata key name in `input_dict` exists in `output_dict`,
            the latter's will get overwritten
        keys_ignore (list):
            The list of keys in `input_dict` to ignore

    Returns:
        dict:
            `output_dict` with the valid `metadata_keys` from `input_dict` copied over
    """

    # get the metadata values to copy over
    metadata_keys = list(input_dict.keys())

    # remove anything set in keys_ignore
    for ki in keys_ignore:
        if ki in input_dict:
            metadata_keys.remove(ki)

    # assign over the remaining metadata keys
    for mk in metadata_keys:
        if type(input_dict[mk]) in [str, int, float, bool, type(None)]:
            output_dict[mk] = input_dict[mk]

    return output_dict


def read_tiling_param(prompt, error_msg, cond, dtype):
    """A helper function to read a tiling param from a user prompt

    Args:
        prompt (str):
            The initial text to display to the user
        error_msg (str):
            The message to display if an invalid input is entered
        cond (function):
            What defines valid input for the variable
        dtype (type):
            The type of variable to read

    Returns:
        Union([int, float, str]):
            The value entered by the user
    """

    # ensure the dtype is valid
    misc_utils.verify_in_list(
        provided_dtype=dtype,
        acceptable_dtypes=[int, float, str]
    )

    while True:
        # read in the variable with correct dtype
        # print error message and re-prompt if cannot be coerced
        try:
            var = dtype(input(prompt))
        except ValueError:
            print(error_msg)
            continue

        # if condition passes, return
        if cond(var):
            return var

        # otherwise, print the error message and re-prompt
        print(error_msg)


def read_fiducial_info():
    """Prompt the user to input the fiducial info (in both stage and optical scoordinates)

    Returns:
        dict:
            Contains the stage and optical coordinates of all 6 required fiducials
    """

    # define the dict to fill in
    fiducial_info = {}

    # store the stage and optical coordinates in separate keys
    fiducial_info['stage'] = {}
    fiducial_info['optical'] = {}

    # read the stage and optical coordinate for each position
    for pos in settings.FIDUCIAL_POSITIONS:
        stage_x = read_tiling_param(
            "Enter the stage x-coordinate of the %s fiducial: " % pos,
            "Error: all fiducial coordinates entered must be positive numbers",
            lambda fc: fc > 0,
            dtype=float
        )

        stage_y = read_tiling_param(
            "Enter the stage y-coordinate of the %s fiducial: " % pos,
            "Error: all fiducial coordinates entered must be positive numbers",
            lambda fc: fc > 0,
            dtype=float
        )

        optical_x = read_tiling_param(
            "Enter the optical x-coordinate of the %s fiducial: " % pos,
            "Error: all fiducial coordinates entered must be positive numbers",
            lambda fc: fc > 0,
            dtype=float
        )

        optical_y = read_tiling_param(
            "Enter the optical y-coordinate of the %s fiducial: " % pos,
            "Error: all fiducial coordinates entered must be positive numbers",
            lambda fc: fc > 0,
            dtype=float
        )

        # define a new stage entry for the fiducial position
        fiducial_info['stage'][pos] = {'x': stage_x, 'y': stage_y}

        # ditto for optical
        fiducial_info['optical'][pos] = {'x': optical_x, 'y': optical_y}

    return fiducial_info


def generate_coreg_params(fiducial_info):
    """Use linear regression from fiducial stage to optical coordinates to define
    co-registration params.

    Separate regressions for x and y values.

    Args:
        fiducial_info (dict):
            The stage and optical coordinates of each fiducial, created by `read_fiducial_info`

    Returns:
        dict:
            Contains the new multiplier and offset along the x- and y-axes
    """

    # define the dict to fill in
    coreg_params = {}

    # extract the data for for x-coordinate stage to optical regression
    x_stage = np.array(
        [fiducial_info['stage'][pos]['x'] for pos in settings.FIDUCIAL_POSITIONS]
    ).reshape(-1, 1)
    x_optical = np.array(
        [fiducial_info['optical'][pos]['x'] for pos in settings.FIDUCIAL_POSITIONS]
    ).reshape(-1, 1)

    # generate x regression params
    x_reg = LinearRegression().fit(x_stage, x_optical)

    # add the multiplier and offset params for x
    x_multiplier = x_reg.coef_[0][0]
    x_offset = x_reg.intercept_[0] / x_multiplier
    coreg_params['STAGE_TO_OPTICAL_X_MULTIPLIER'] = x_multiplier
    coreg_params['STAGE_TO_OPTICAL_X_OFFSET'] = x_offset

    # extract the data for for y-coordinate stage to optical regression
    y_stage = np.array(
        [fiducial_info['stage'][pos]['y'] for pos in settings.FIDUCIAL_POSITIONS]
    ).reshape(-1, 1)
    y_optical = np.array(
        [fiducial_info['optical'][pos]['y'] for pos in settings.FIDUCIAL_POSITIONS]
    ).reshape(-1, 1)

    # generate y regression params
    y_reg = LinearRegression().fit(y_stage, y_optical)

    # add the multiplier and offset params for y
    y_multiplier = y_reg.coef_[0][0]
    y_offset = y_reg.intercept_[0] / y_multiplier
    coreg_params['STAGE_TO_OPTICAL_Y_MULTIPLIER'] = y_multiplier
    coreg_params['STAGE_TO_OPTICAL_Y_OFFSET'] = y_offset

    return coreg_params


def save_coreg_params(coreg_params):
    """Save the co-registration parameters to `coreg_params.json` in `toffy`

    Args:
        coreg_params (dict):
            Contains the multiplier and offsets for co-registration along the x- and y-axis
    """

    # generate the time this set of co-registration parameters were generated
    coreg_params['date'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    # write to a new coreg_params.json file if it doesn't already exist
    if not os.path.exists(os.path.join('..', 'toffy', 'coreg_params.json')):
        coreg_data = {
            'coreg_params': [
                coreg_params
            ]
        }

        with open(os.path.join('..', 'toffy', 'coreg_params.json'), 'w') as cp:
            json.dump(coreg_data, cp)
    # append to the existing coreg_params key if coreg_params.json already exists
    else:
        with open(os.path.join('..', 'toffy', 'coreg_params.json'), 'r') as cp:
            coreg_data = json.load(cp)

        coreg_data['coreg_params'].append(coreg_params)

        with open(os.path.join('..', 'toffy', 'coreg_params.json'), 'w') as cp:
            json.dump(coreg_data, cp)


def generate_region_info(region_params):
    """Generate the `region_params` list in the tiling parameter dict

    Args:
        region_params (dict):
            A `dict` mapping each region-specific parameter to a list of values per FOV

    Returns:
        list:
            The complete set of `region_params` sorted by region
    """

    # define the region params list
    region_params_list = []

    # iterate over all the region parameters, all parameter lists are the same length
    for i in range(len(region_params['region_start_row'])):
        # define a dict containing all the region info for the specific FOV
        region_info = {
            rp: region_params[rp][i] for rp in region_params
        }

        # append info to region_params
        region_params_list.append(region_info)

    return region_params_list


def read_tiled_region_inputs(region_corners, region_params):
    """Reads input for tiled regions from user and `region_corners`.

    Updates all the tiling params inplace. Units used are microns.

    Args:
        region_corners (dict):
            The data containing the FOVs used to define the upper-left corner of each tiled region
        region_params (dict):
            A `dict` mapping each region-specific parameter to a list of values per FOV
    """

    # read in the data for each region (region_start from region_corners_path, others from user)
    for fov in region_corners['fovs']:
        # append the name of the region
        region_params['region_name'].append(fov['name'])

        # append the starting row and column coordinates
        region_params['region_start_row'].append(fov['centerPointMicrons']['y'])
        region_params['region_start_col'].append(fov['centerPointMicrons']['x'])

        print("Using start coordinates of (%d, %d) in microns for region %s"
              % (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'], fov['name']))

        # verify that the micron size specified is valid
        if fov['fovSizeMicrons'] <= 0:
            raise ValueError("The fovSizeMicrons field for FOVs in region %s must be positive"
                             % fov['name'])

        print("Using FOV step size of %d microns for both row (y) and column (x) axis of region %s"
              % (fov['fovSizeMicrons'], fov['name']))

        # use fovSizeMicrons as the step size along both axes
        region_params['row_fov_size'].append(fov['fovSizeMicrons'])
        region_params['col_fov_size'].append(fov['fovSizeMicrons'])

        # allow the user to specify the number of fovs along each dimension
        num_row = read_tiling_param(
            "Enter the number of FOVs per row for region %s: " % fov['name'],
            "Error: number of FOVs per row must be a positive integer",
            lambda nx: nx >= 1,
            dtype=int
        )

        num_col = read_tiling_param(
            "Enter the number of FOVs per column for region %s: " % fov['name'],
            "Error: number of FOVs per column must be a positive integer",
            lambda ny: ny >= 1,
            dtype=int
        )

        region_params['fov_num_row'].append(num_row)
        region_params['fov_num_col'].append(num_col)

        # allow the user to specify if the FOVs should be randomized
        randomize = read_tiling_param(
            "Randomize FOVs for region %s? Y/N: " % fov['name'],
            "Error: randomize parameter must Y or N",
            lambda r: r in ['Y', 'N', 'y', 'n'],
            dtype=str
        )

        randomize = randomize.upper()

        region_params['region_rand'].append(randomize)


def set_tiled_region_params(region_corners_path):
    """Given a file specifying top-left FOVs for a set of regions, set the MIBI tiling parameters.

    User inputs will be required for many values. Units used are microns.

    Args:
        region_corners_path (str):
            Path to the JSON file containing the FOVs used to define the upper-left corner
            of each tiled region

    Returns:
        dict:
            Contains the tiling parameters for each tiled region
    """

    # file path validation
    if not os.path.exists(region_corners_path):
        raise FileNotFoundError(
            "Tiled region corners list file %s does not exist" % region_corners_path
        )

    # read in the region corners data
    with open(region_corners_path, 'r', encoding='utf-8') as flf:
        tiled_region_corners = json.load(flf)
    tiled_region_corners = json_utils.rename_missing_fovs(tiled_region_corners)

    # define the parameter dict to return
    tiling_params = {}

    # copy over the metadata values from tiled_region_corners to tiling_params
    tiling_params = assign_metadata_vals(tiled_region_corners, tiling_params, ['fovs'])

    # define the region_params dict
    region_params = {rpf: [] for rpf in settings.REGION_PARAM_FIELDS}

    # prompt the user for params associated with each tiled region
    read_tiled_region_inputs(tiled_region_corners, region_params)

    # need to copy fov metadata over, needed for generate_fov_list
    tiling_params['fovs'] = copy.deepcopy(tiled_region_corners['fovs'])

    # store the read in parameters in the region_params key
    tiling_params['region_params'] = generate_region_info(region_params)

    # whether to insert moly points between regions
    moly_region_insert = read_tiling_param(
        "Insert a moly point between each tiled region? \
            If yes, you must provide a path to the example moly_FOV json file. Y/N: ",
        "Error: moly point region parameter must be either Y or N",
        lambda mri: mri in ['Y', 'N', 'y', 'n'],
        dtype=str
    )

    # convert to uppercase to standardize
    moly_region_insert = moly_region_insert.upper()
    tiling_params['moly_region'] = moly_region_insert

    # whether to insert moly points between fovs
    moly_interval = read_tiling_param(
        "Enter the FOV interval size to insert Moly points. If yes, you must provide \
            a path to the example moly_FOV json file and enter the number of FOVs "
        "between each Moly point. If no, enter 0: ",
        "Error: moly interval must be 0 or a positive integer",
        lambda mi: mi >= 0,
        dtype=int
    )

    if moly_interval > 0:
        tiling_params['moly_interval'] = moly_interval

    return tiling_params


def generate_x_y_fov_pairs(x_range, y_range):
    """Given all x and y coordinates (in microns) a FOV can take,
    generate all possible `(x, y)` pairings

    Args:
        x_range (list):
            Range of x values a FOV can take
        y_range (list):
            Range of y values a FOV can take

    Returns:
        list:
            Every possible `(x, y)` pair for a FOV
    """

    # define a list to hold all the (x, y) pairs
    all_pairs = []

    # iterate over all combinations of x and y
    for t in combinations((x_range, y_range), 2):
        # compute the product of the resulting x and y list pair, append results
        for pair in product(t[0], t[1]):
            all_pairs.append(pair)

    return all_pairs


def generate_x_y_fov_pairs_rhombus(top_left, top_right, bottom_left, bottom_right,
                                   num_row, num_col):
    """Generates coordinates (in microns) of FOVs as defined by corners of a rhombus

    Args:
        top_left (XYCoord): coordinate of top left corner
        top_right (XYCoord): coordinate of top right corner
        bottom_left (XYCoord): coordinate of bottom right corner
        bottom_right (XYCoord): coordiante of bottom right corner
        num_row (int): number of fovs on row dimension
        num_col (int): number of fovs on column dimension

    Returns:
        list: coordinates for all FOVs defined by region"""

    # compute the vertical shift in the top and bottom row of the TMA
    top_row_shift = top_right.y - top_left.y
    bottom_row_shift = bottom_right.y - bottom_left.y

    # average between the two will be used to increment indices
    avg_row_shift = (top_row_shift + bottom_row_shift) / 2

    # compute horizontal shift in the left and right column of the TMA
    left_col_shift = bottom_left.x - top_left.x
    right_col_shift = bottom_right.x - top_right.x

    # average between the two will be used to increment indices
    avg_col_shift = (left_col_shift + right_col_shift) / 2

    # compute per-FOV adjustment
    row_increment = avg_row_shift / (num_col - 1)
    col_increment = avg_col_shift / (num_row - 1)

    # compute baseline indices for a rectangle with same coords
    row_dif = bottom_left.y - top_left.y
    col_dif = top_right.x - top_left.x

    row_baseline = row_dif / (num_row - 1)
    col_baseline = col_dif / (num_col - 1)

    pairs = []

    for i in range(num_col):
        for j in range(num_row):
            x_coord = top_left.x + col_baseline * i + col_increment * j
            y_coord = top_left.y + row_baseline * j + row_increment * i
            pairs.append((int(x_coord), int(y_coord)))

    return pairs


def generate_tiled_region_fov_list(tiling_params, moly_path: Optional[str] = None):
    """Generate the list of FOVs on the image from the `tiling_params` set for tiled regions

    Moly point insertion: happens once every number of FOVs you specified in
    `tiled_region_set_params`. There are a couple caveats to keep in mind:

    - The interval specified will not reset between regions. In other words, if the interval is 3
      and the next set of FOVs contains 2 in region 1 and 1 in region 2, the next Moly point will
      be placed after the 1st FOV in region 2 (not after the 3rd FOV in region 2). Moly points
      inserted between regions are ignored in this calculation.
    - If the interval specified cleanly divides the number of FOVs in a region, a Moly point will
      not be placed at the end of the region. Suppose 3 FOVs are defined along both the x- and
      y-axis for region 1 (for a total of 9 FOVs) and a Moly point FOV interval of 3 is specified.
      Without also setting Moly point insertion between different regions, a Moly point will NOT be
      placed after the last FOV of region 1 (the next Moly point will appear after the 3rd
      FOV in in region 2).

    Args:
        tiling_params (dict):
            The tiling parameters created by `set_tiled_region_params`
        moly_path (Optional[str]):
            The path to the Moly point to insert between FOV intervals and/or regions.
            If these insertion parameters are not specified in `tiling_params`, this won't be used.
            Defaults to None.

    Returns:
        dict:
            Data containing information about each FOV
    """

    # file path validation
    if (tiling_params.get("moly_region", "N") == "Y") or \
       (tiling_params.get("moly_interval", 0) > 0):
        if not os.path.exists(moly_path):
            raise FileNotFoundError("The provided Moly FOV file %s does not exist. If you want\
                                    to include Moly FOVs you must provide a valid path. Otherwise\
                                    , select 'No' for the options relating to Moly FOVs"
                                    % moly_path)

        # read in the moly point data
        with open(moly_path, 'r', encoding='utf-8') as mpf:
            moly_point = json.load(mpf)

    # define the fov_regions dict
    fov_regions = {}

    # copy over the metadata values from tiling_params to fov_regions
    fov_regions = assign_metadata_vals(
        tiling_params, fov_regions, ['region_params', 'moly_region', 'moly_interval']
    )

    # define a specific FOVs field in fov_regions, this will contain the actual FOVs
    fov_regions['fovs'] = []

    # define a counter to determine where to insert a moly point
    # only used if moly_interval is set in tiling_params
    # NOTE: total_fovs is used to prevent moly_counter from initiating the addition of
    # a Moly point at the end of a region
    moly_counter = 0
    total_fovs = 0

    # iterate through each region and append created fovs to fov_regions['fovs']
    for region_index, region_info in enumerate(tiling_params['region_params']):
        # extract start coordinates
        start_row = region_info['region_start_row']
        start_col = region_info['region_start_col']

        # define the range of x- and y-coordinates to use
        row_range = list(range(region_info['fov_num_row']))
        col_range = list(range(region_info['fov_num_col']))

        # create all pairs between two lists
        row_col_pairs = generate_x_y_fov_pairs(row_range, col_range)

        # name the FOVs according to MIBI conventions
        fov_names = ['%s_R%dC%d' % (region_info['region_name'], y + 1, x + 1)
                     for x in range(region_info['fov_num_row'])
                     for y in range(region_info['fov_num_col'])]

        # randomize pairs list if specified
        if region_info['region_rand'] == 'Y':
            # make sure the fov_names are set in the same shuffled indices for renaming
            row_col_pairs, fov_names = shuffle(row_col_pairs, fov_names)

        # update total_fovs, we'll prevent moly_counter from triggering the appending of
        # a Moly point at the end of a region this way
        total_fovs += len(row_col_pairs)

        for index, (col_i, row_i) in enumerate(row_col_pairs):
            # use the fov size to scale to the current x- and y-coordinate
            cur_row = start_row - row_i * region_info['row_fov_size']
            cur_col = start_col + col_i * region_info['col_fov_size']

            # copy the fov metadata over and add cur_x, cur_y, and name
            fov = copy.deepcopy(tiling_params['fovs'][region_index])
            fov['centerPointMicrons']['x'] = cur_col
            fov['centerPointMicrons']['y'] = cur_row
            fov['name'] = fov_names[index]

            # append value to fov_regions
            fov_regions['fovs'].append(fov)

            # increment moly_counter as we've added another fov
            moly_counter += 1

            # append a Moly point if moly_interval is set and we've reached the interval threshold
            # the exception: don't insert a Moly point at the end of a region
            if 'moly_interval' in tiling_params and \
               moly_counter % tiling_params['moly_interval'] == 0 and \
               moly_counter < total_fovs:
                fov_regions['fovs'].append(moly_point)

        # append Moly point to seperate regions if not last and if specified
        if 'moly_region' in tiling_params and \
            tiling_params['moly_region'] == 'Y' and \
           region_index != len(tiling_params['region_params']) - 1:
            fov_regions['fovs'].append(moly_point)

    return fov_regions


def validate_tma_corners(top_left, top_right, bottom_left, bottom_right):
    """Ensures that the provided TMA corners match assumptions

    Args:
        top_left (XYCoord): coordinate (in microns) of top left corner
        top_right (XYCoord): coordinate (in microns) of top right corner
        bottom_left (XYCoord): coordinate (in microns) of bottom right corner
        bottom_right (XYCoord): coordinate (in microns)of bottom right corner

    """
    # TODO: should we programmatically validate all pairwise comparisons?

    if top_left.x > top_right.x:
        raise ValueError("Invalid corner file: The upper left corner is "
                         "to the right of the upper right corner")

    if bottom_left.x > bottom_right.x:
        raise ValueError("Invalid corner file: The bottom left corner is "
                         "to the right of the bottom right corner")

    if top_left.y < bottom_left.y:
        raise ValueError("Invalid corner file: The upper left corner is "
                         "below the bottom left corner")

    if top_right.y < bottom_right.y:
        raise ValueError("Invalid corner file: The upper right corner is "
                         "below the bottom right corner")


@dataclass
class XYCoord:
    x: float
    y: float


def generate_tma_fov_list(tma_corners_path, num_fov_row, num_fov_col):
    """Generate the list of FOVs on the image using the TMA input file in `tma_corners_path`

    NOTE: unlike tiled regions, the returned list of FOVs is just an intermediate step to the
    interactive remapping process. So the result will just be each FOV name mapped to its centroid
    (in microns).

    Args:
        tma_corners_path (dict):
            Path to the JSON file containing the FOVs used to define the tiled TMA region
        num_fov_row (int):
            Number of FOVs along the row dimension
        num_fov_col (int):
            Number of FOVs along the column dimension

    Returns:
        dict:
            Data containing information about each FOV (just FOV name mapped to centroid)
    """

    # file path validation
    if not os.path.exists(tma_corners_path):
        raise FileNotFoundError(
            "TMA corners file %s does not exist" % tma_corners_path
        )

    # user needs to define at least 3 FOVs along the x- and y-axes
    if num_fov_row < 3:
        raise ValueError("Number of TMA-grid rows must be at least 3")

    if num_fov_col < 3:
        raise ValueError("Number of TMA-grid columns must be at least 3")

    # read in tma_corners_path
    with open(tma_corners_path, 'r', encoding='utf-8') as flf:
        tma_corners = json.load(flf)
    tma_corners = json_utils.rename_missing_fovs(tma_corners)

    # a TMA can only be defined by four FOVs, one for each corner
    if len(tma_corners['fovs']) != 4:
        raise ValueError("Your FOV region file %s needs to contain four FOVs" % tma_corners)

    # retrieve the FOVs from JSON file
    corners = [0] * 4
    for i, fov in enumerate(tma_corners['fovs']):
        corners[i] = XYCoord(*itemgetter('x', 'y')(fov['centerPointMicrons']))

    top_left, top_right, bottom_left, bottom_right = corners

    validate_tma_corners(top_left, top_right, bottom_left, bottom_right)

    # create all x_y coordinates
    x_y_pairs = generate_x_y_fov_pairs_rhombus(top_left, top_right, bottom_left, bottom_right,
                                               num_fov_row, num_fov_col)

    # name the FOVs according to MIBI conventions
    fov_names = ['R%dC%d' % (y + 1, x + 1) for x in range(num_fov_col) for y in range(num_fov_row)]

    # define the fov_regions dict
    fov_regions = {}

    # map each name to its corresponding coordinate value
    for index, (xi, yi) in enumerate(x_y_pairs):
        fov_regions[fov_names[index]] = (xi, yi)

    return fov_regions


def convert_stage_to_optical(coord, stage_optical_coreg_params):
    """Convert the coordinate in stage microns to optical pixels.

    In other words, co-register using the centroid of a FOV.

    The values are coerced to ints to allow indexing into the slide.
    Coordinates are returned in `(y, x)` form to account for a different coordinate axis.

    Args:
        coord (tuple):
            The coordinate in microns to convert
        stage_optical_coreg_params (dict):
            Contains the co-registration parameters to use

    Returns:
        tuple:
            The converted coordinate from stage microns to optical pixels.
            Values truncated to `int`.
    """

    stage_to_optical_x_multiplier = stage_optical_coreg_params['STAGE_TO_OPTICAL_X_MULTIPLIER']
    stage_to_optical_x_offset = stage_optical_coreg_params['STAGE_TO_OPTICAL_X_OFFSET']
    stage_to_optical_y_multiplier = stage_optical_coreg_params['STAGE_TO_OPTICAL_Y_MULTIPLIER']
    stage_to_optical_y_offset = stage_optical_coreg_params['STAGE_TO_OPTICAL_Y_OFFSET']

    # NOTE: all conversions are done using the fiducials
    # convert from microns to stage coordinates
    stage_coord_x = (
        coord[0] * settings.MICRON_TO_STAGE_X_MULTIPLIER - settings.MICRON_TO_STAGE_X_OFFSET
    )
    stage_coord_y = (
        coord[1] * settings.MICRON_TO_STAGE_Y_MULTIPLIER - settings.MICRON_TO_STAGE_Y_OFFSET
    )

    # convert from stage coordinates to optical pixels
    pixel_coord_x = (stage_coord_x + stage_to_optical_x_offset) * stage_to_optical_x_multiplier
    pixel_coord_y = (stage_coord_y + stage_to_optical_y_offset) * stage_to_optical_y_multiplier

    return (int(pixel_coord_y), int(pixel_coord_x))


def assign_closest_fovs(manual_fovs, auto_fovs):
    """For each FOV in `manual_fovs`, map it to its closest FOV in `auto_fovs`

    Args:
        manual_fovs (dict):
            The list of FOVs proposed by the user
        auto_fovs (dict):
            The list of FOVs generated by `set_tiling_params` in `example_fov_grid_generate.ipynb`

    Returns:
        tuple:

        - A `dict` mapping each manual FOV to an auto FOV and its respective distance
          (in microns) from it
        - A `pandas.DataFrame` defining the distance (in microns) from each manual FOV
          to each auto FOV, row indices are manual FOVs, column names are auto FOVs
    """

    # condense the manual FOVs JSON list into just a mapping between name and coordinate
    manual_fovs_name_coord = {
        fov['name']: tuple(list(fov['centerPointMicrons'].values()))

        for fov in manual_fovs['fovs']
    }

    # retrieve the centroids in array format for distance calculation
    manual_centroids = np.array(
        [list(centroid) for centroid in manual_fovs_name_coord.values()]
    )

    auto_centroids = np.array(
        [list(centroid) for centroid in auto_fovs.values()]
    )

    # define the mapping dict from manual to auto
    manual_to_auto_map = {}

    # compute the euclidean distances between the manual and the auto centroids
    manual_auto_dist = np.linalg.norm(
        manual_centroids[:, np.newaxis] - auto_centroids, axis=2
    )

    # for each manual fov, get the index of the auto fov closest to it
    closest_auto_point_ind = np.argmin(manual_auto_dist, axis=1)

    # assign the mapping in manual_to_auto_map
    for manual_index, auto_index in enumerate(closest_auto_point_ind):
        # get the coordinates of the manual fov and its closest auto fov
        manual_coords = tuple(manual_centroids[manual_index])
        auto_coords = tuple(auto_centroids[auto_index])

        # get the corresponding fov names
        man_name = list(manual_fovs_name_coord.keys())[manual_index]
        auto_name = list(auto_fovs.keys())[auto_index]

        # map the manual fov name to its closest auto fov name
        manual_to_auto_map[man_name] = auto_name

    # convert manual_auto_dist into a Pandas DataFrame, this makes it easier to index
    manual_auto_dist = pd.DataFrame(
        manual_auto_dist,
        index=list(manual_fovs_name_coord.keys()),
        columns=list(auto_fovs.keys())
    )

    return manual_to_auto_map, manual_auto_dist


def generate_fov_circles(manual_fovs_info, auto_fovs_info,
                         manual_name, auto_name, slide_img, draw_radius=7):
    """Draw the circles defining each FOV (manual and auto) on `slide_img`

    Args:
        manual_fovs_info (dict):
            maps each manual FOV to its centroid coordinates (in optical pixels)
        auto_fovs_info (dict):
            maps each auto FOV to its centroid coordinates (in optical pixels)
        manual_name (str):
            the name of the manual FOV to highlight
        auto_name (str):
            the name of the automatically-generated FOV to highlight
        slide_img (numpy.ndarray):
            the image to overlay
        draw_radius (int):
            the radius (in optical pixels) of the circle to overlay for each FOV

    Returns:
        numpy.ndarray:
            `slide_img` with defining each manually-defined and automatically-generated FOV
    """

    # define dicts to hold the coordinates
    manual_coords = {}
    auto_coords = {}

    # define the fov size boundaries, needed to prevent drawing a circle out of bounds
    fov_size = slide_img.shape[:2]

    # generate the regions for each manual and mapped auto fov
    for mfi in manual_fovs_info:
        # get the x- and y-coordinate of the centroid
        manual_x = int(manual_fovs_info[mfi][0])
        manual_y = int(manual_fovs_info[mfi][1])

        # define the circle coordinates for the region
        mr_x, mr_y = ellipse(
            manual_x, manual_y, draw_radius, draw_radius, shape=fov_size
        )

        # color the selected manual fov dark red, else bright red
        if mfi == manual_name:
            slide_img[mr_x, mr_y, 0] = 210
            slide_img[mr_x, mr_y, 1] = 37
            slide_img[mr_x, mr_y, 2] = 37
        else:
            slide_img[mr_x, mr_y, 0] = 255
            slide_img[mr_x, mr_y, 1] = 133
            slide_img[mr_x, mr_y, 2] = 133

    # repeat but for the automatically generated points
    for afi in auto_fovs_info:
        # repeat the above for auto points
        auto_x = int(auto_fovs_info[afi][0])
        auto_y = int(auto_fovs_info[afi][1])

        # define the circle coordinates for the region
        ar_x, ar_y = ellipse(
            auto_x, auto_y, draw_radius, draw_radius, shape=fov_size
        )

        # color the selected auto fov dark blue, else bright blue
        if afi == auto_name:
            slide_img[ar_x, ar_y, 0] = 50
            slide_img[ar_x, ar_y, 1] = 115
            slide_img[ar_x, ar_y, 2] = 229
        else:
            slide_img[ar_x, ar_y, 0] = 162
            slide_img[ar_x, ar_y, 1] = 197
            slide_img[ar_x, ar_y, 2] = 255

    return slide_img


def update_mapping_display(change, w_auto, manual_to_auto_map, manual_coords, auto_coords,
                           slide_img, draw_radius=7):
    """Changes the highlighted manual-auto fov pair on the image based on new selected manual FOV

    Helper to `update_mapping` nested callback function in `interactive_remap`

    Args:
        change (dict):
            defines the properties of the changed value of the manual FOV menu
        w_auto (ipywidgets.widgets.widget_selection.Dropdown):
            the dropdown menu handler for the automatically-generated FOVs
        manual_to_auto_map (dict):
            defines the mapping of manual to auto FOV names
        manual_coords (dict):
            maps each manually-defined FOV to its coordinate (in optical pixels)
        auto_coords (dict):
            maps each automatically-generated FOV to its coordinate (in optical pixels)
        slide_img (numpy.ndarray):
            the image to overlay
        draw_radius (int):
            the radius (in optical pixels) to draw each circle on the slide

    Returns:
        numpy.ndarray:
            `slide_img` with the updated circles after manual fov changed
    """

    # define the fov size boundaries, needed to prevent drawing a circle out of bounds
    fov_size = slide_img.shape[:2]

    # retrieve the old manual centroid
    old_manual_x, old_manual_y = manual_coords[change['old']]

    # redraw the old manual centroid on the slide_img
    old_mr_x, old_mr_y = ellipse(
        old_manual_x, old_manual_y, draw_radius, draw_radius, shape=fov_size
    )

    slide_img[old_mr_x, old_mr_y, 0] = 255
    slide_img[old_mr_x, old_mr_y, 1] = 133
    slide_img[old_mr_x, old_mr_y, 2] = 133

    # retrieve the old auto centroid
    old_auto_x, old_auto_y = auto_coords[w_auto.value]

    # redraw the old auto centroid on the slide_img
    old_ar_x, old_ar_y = ellipse(
        old_auto_x, old_auto_y, draw_radius, draw_radius, shape=fov_size
    )

    slide_img[old_ar_x, old_ar_y, 0] = 162
    slide_img[old_ar_x, old_ar_y, 1] = 197
    slide_img[old_ar_x, old_ar_y, 2] = 255

    # retrieve the new manual centroid
    new_manual_x, new_manual_y = manual_coords[change['new']]

    # redraw the new manual centroid on the slide_img
    new_mr_x, new_mr_y = ellipse(
        new_manual_x, new_manual_y, draw_radius, draw_radius, shape=fov_size
    )

    slide_img[new_mr_x, new_mr_y, 0] = 210
    slide_img[new_mr_x, new_mr_y, 1] = 37
    slide_img[new_mr_x, new_mr_y, 2] = 37

    # retrieve the new auto centroid
    new_auto_x, new_auto_y = auto_coords[manual_to_auto_map[change['new']]]

    # redraw the new auto centroid on the slide_img
    new_ar_x, new_ar_y = ellipse(
        new_auto_x, new_auto_y, draw_radius, draw_radius, shape=fov_size
    )

    slide_img[new_ar_x, new_ar_y, 0] = 50
    slide_img[new_ar_x, new_ar_y, 1] = 115
    slide_img[new_ar_x, new_ar_y, 2] = 229

    # set the mapped auto value according to the new manual value
    w_auto.value = manual_to_auto_map[change['new']]

    return slide_img


def remap_manual_to_auto_display(change, w_man, manual_to_auto_map, manual_auto_dist,
                                 auto_coords, slide_img, draw_radius=7,
                                 check_dist=2000, check_duplicates=True, check_mismatches=True):
    """changes the highlighted automatically-generated FOV to new selected auto FOV
    and updates the mapping in `manual_to_auto_map`

    Helper to `remap_values` nested callback function in `interactive_remap`

    Args:
        change (dict):
            defines the properties of the changed value of the automatically-generated FOV menu
        w_man (ipywidgets.widgets.widget_selection.Dropdown):
            the dropdown menu handler for the manual FOVs
        manual_to_auto_map (dict):
            defines the mapping of manual to auto FOV names
        manual_auto_dist (pandas.DataFrame):
            defines the distance (in microns) between each manual FOV from each auto FOV
        auto_coords (dict):
            maps each automatically-generated FOV to its coordinate (in optical pixels)
        slide_img (numpy.ndarray):
            the image to overlay
        draw_radius (int):
            the radius (in optical pixels) to draw each circle on the slide
        check_dist (float):
            if the distance (in microns) between a manual-auto FOV pair exceeds this value, it will
            be reported for a potential error, if `None` does not validate distance
        check_duplicates (bool):
            if `True`, validate whether an auto FOV has 2 manual FOVs mapping to it
        check_mismatches (bool):
            if `True`, validate whether the the manual auto FOV pairs have matching names

    Returns:
        tuple:
            contains the following elements

            - `numpy.ndarray`:`slide_img` with the updated circles after auto fov changed
              remapping the fovs
            - `str`: the new error message to display after a remapping
    """

    # define the fov size boundaries, needed to prevent drawing a circle out of bounds
    fov_size = slide_img.shape[:2]

    # retrieve the coordinates for the old auto centroid w_prop mapped to
    old_auto_x, old_auto_y = auto_coords[change['old']]

    # redraw the old auto centroid on the slide_img
    old_ar_x, old_ar_y = ellipse(
        old_auto_x, old_auto_y, draw_radius, draw_radius, shape=fov_size
    )

    slide_img[old_ar_x, old_ar_y, 0] = 162
    slide_img[old_ar_x, old_ar_y, 1] = 197
    slide_img[old_ar_x, old_ar_y, 2] = 255

    # retrieve the coordinates for the new auto centroid w_prop maps to
    new_auto_x, new_auto_y = auto_coords[change['new']]

    # redraw the new auto centroid on the slide_img
    new_ar_x, new_ar_y = ellipse(
        new_auto_x, new_auto_y, draw_radius, draw_radius, shape=fov_size
    )

    slide_img[new_ar_x, new_ar_y, 0] = 50
    slide_img[new_ar_x, new_ar_y, 1] = 115
    slide_img[new_ar_x, new_ar_y, 2] = 229

    # remap the manual fov to the changed value
    manual_to_auto_map[w_man.value] = change['new']

    # define the potential sources of error in the new mapping
    manual_auto_warning = generate_validation_annot(
        manual_to_auto_map, manual_auto_dist, check_dist, check_duplicates, check_mismatches
    )

    return slide_img, manual_auto_warning


def write_manual_to_auto_map(manual_to_auto_map, save_ann, mapping_path):
    """Saves the manually-defined to automatically-generated FOV map and notifies the user

    Helper to `save_mapping` nested callback function in `interactive_remap`

    Args:
        manual_to_auto_map (dict):
            defines the mapping of manual to auto FOV names
        save_ann (dict):
            contains the annotation object defining the save notification
        mapping_path (str):
            the path to the file to save the mapping to
    """

    # save the mapping
    with open(mapping_path, 'w', encoding='utf-8') as mp:
        json.dump(manual_to_auto_map, mp)

    # remove the save annotation if it already exists
    # clears up some space if the user decides to save several times
    if save_ann['annotation']:
        save_ann['annotation'].remove()

    # get the current datetime, need to display when the annotation was saved
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # display save annotation above the plot
    save_msg = plt.annotate(
        'Mapping saved at %s!' % timestamp,
        (0, 20),
        color='white',
        fontweight='bold',
        annotation_clip=False
    )

    # assign annotation to save_ann
    save_ann['annotation'] = save_msg


# TODO: potential type hinting candidate?
def find_manual_auto_invalid_dist(manual_to_auto_map, manual_auto_dist, dist_threshold=2000):
    """Finds the manual FOVs that map to auto FOVs greater than `dist_threshold` away (in microns)

    Args:
        manual_to_auto_map (dict):
            defines the mapping of manual to auto FOV names
        manual_auto_dist (pandas.DataFrame):
            defines the distance (in microns) between each manual FOV from each auto FOV
        dist_threshold (float):
            if the distance (in microns) between a manual-auto FOV pair exceeds this value, it will
            be reported for a potential error

    Returns:
        list:
            contains tuples with elements:

            - `str`: the manual FOV name
            - `str`: the auto FOV name
            - `float`: the distance (in microns) between the manual and auto FOV

            applies only for manual-auto pairs with distance greater than `dist_threshold`
            (in microns), sorted by decreasing manual-auto FOV distance
    """

    # define the fov pairs at a distance greater than dist_thresh
    manual_auto_invalid_dist_pairs = [
        (mf, af, manual_auto_dist.loc[mf, af])
        for (mf, af) in manual_to_auto_map.items()
        if manual_auto_dist.loc[mf, af] > dist_threshold
    ]

    # sort these fov pairs by distance descending
    manual_auto_invalid_dist_pairs = sorted(
        manual_auto_invalid_dist_pairs, key=lambda val: val[2], reverse=True
    )

    return manual_auto_invalid_dist_pairs


def find_duplicate_auto_mappings(manual_to_auto_map):
    """Finds each auto FOV with more than one manual FOV mapping to it

    Args:
        manual_to_auto_map (dict):
            defines the mapping of manual to auto FOV names

    Returns:
        list:
            contains tuples with elements:

            - `str`: the name of the auto FOV
            - `tuple`: the set of manual FOVs that map to the auto FOV

            only for auto FOVs with more than one manual FOV mapping to it
    """

    # "reverse" manual_to_auto_map: for each auto FOV find the list of manual FOVs that map to it
    auto_fov_mappings = {}

    # good ol' incremental dict building!
    for mf in manual_to_auto_map:
        closest_auto_fov = manual_to_auto_map[mf]

        if closest_auto_fov not in auto_fov_mappings:
            auto_fov_mappings[closest_auto_fov] = []

        auto_fov_mappings[closest_auto_fov].append(mf)

    # only keep the auto FOVs with more than one manual FOV mapping to it
    duplicate_auto_fovs = [
        (af, tuple(mf_list)) for (af, mf_list) in auto_fov_mappings.items() if len(mf_list) > 1
    ]

    # sort auto FOVs alphabetically
    duplicate_auto_fovs = sorted(duplicate_auto_fovs, key=lambda val: val[0])

    return duplicate_auto_fovs


def find_manual_auto_name_mismatches(manual_to_auto_map):
    """Finds the manual FOVs with names that do not match their corresponding auto FOV

    Args:
        manual_to_auto_map (dict):
            defines the mapping of manual to auto FOV names

    Returns:
        list:
            contains tuples with elements:

            - `str`: the manual FOV
            - `str`: the corresponding auto FOV
    """

    # find the manual FOVs that don't match their corresponding closest_auto_fov name
    # NOTE: this method maintains the original manual FOV ordering which is already sorted
    manual_auto_mismatches = [
        (k, v)
        for (k, v) in manual_to_auto_map.items() if k != v
    ]

    return manual_auto_mismatches


# TODO: potential type hinting candidate?
def generate_validation_annot(manual_to_auto_map, manual_auto_dist, check_dist=2000,
                              check_duplicates=True, check_mismatches=True):
    """Finds problematic manual-auto FOV pairs and generates a warning message to display

    The following potential sources of error can be requested by the user:

    - Manual to auto FOV pairs that are of a distance greater than `check_dist` away (in microns)
    - Auto FOV names that have more than one manual FOV name mapping to it
    - Manual to auto FOV pairs that don't have the same name

    Args:
        manual_to_auto_map (dict):
            defines the mapping of manual to auto FOV names
        manual_auto_dist (pandas.DataFrame):
            defines the distance (in microns) between each manual FOV from each auto FOV
        check_dist (float):
            if the distance (in microns) between a manual-auto FOV pair exceeds this value, it will
            be reported for a potential error, if `None` does not validate distance
        check_duplicates (bool):
            if `True`, report each auto FOV with 2 or more manual FOVs mapping to it
        check_mismatches (bool):
            if `True`, report manual FOVs that map to an auto FOV with a different name

    Returns:
        str:
            describes the validation failures
    """

    # define the potential sources of error desired by user in the mapping
    invalid_dist = find_manual_auto_invalid_dist(
        manual_to_auto_map, manual_auto_dist, check_dist
    ) if check_dist is not None else []
    duplicate_auto = find_duplicate_auto_mappings(
        manual_to_auto_map
    ) if check_duplicates else []
    name_mismatch = find_manual_auto_name_mismatches(
        manual_to_auto_map
    ) if check_mismatches else []

    # generate the annotation
    warning_annot = ""

    # add the manual-auto FOV pairs with distances greater than check_dist
    if len(invalid_dist) > 0:
        warning_annot += \
            'The following mappings are placed more than %d microns apart:\n\n' % check_dist

        warning_annot += '\n'.join([
            'User-defined FOV %s to TMA-grid FOV %s (distance: %.2f)' % (mf, af, dist)
            for (mf, af, dist) in invalid_dist
        ])

        warning_annot += '\n\n'

    # add the auto FOVs that have more than one manual FOV mapping to it
    if len(duplicate_auto) > 0:
        warning_annot += \
            'The following TMA-grid FOVs have more than one user-defined FOV mapping to it:\n\n'

        warning_annot += '\n'.join([
            'TMA-grid FOV %s: mapped with user-defined FOVs %s' % (af, ', '.join(mf_tuple))
            for (af, mf_tuple) in duplicate_auto
        ])

        warning_annot += '\n\n'

    # add the manual-auto FOV pairs with mismatched names
    if len(name_mismatch) > 0:
        warning_annot += \
            'The following mappings have mismatched names:\n\n'

        warning_annot += '\n'.join([
            'User-defined FOV %s: mapped with TMA-grid FOV %s' % (mf, af)
            for (mf, af) in name_mismatch
        ])

        warning_annot += '\n\n'

    return warning_annot


# TODO: potential type hinting candidate?
def tma_interactive_remap(manual_fovs, auto_fovs, slide_img, mapping_path,
                          draw_radius=7, figsize=(7, 7),
                          check_dist=2000, check_duplicates=True, check_mismatches=True):
    """Creates the remapping interactive interface for manual to auto FOVs

    Args:
        manual_fovs (dict):
            The list of FOVs proposed by the user
        auto_fovs (dict):
            The list of FOVs created by `generate_tma_fov_list` run in `autolabel_tma_cores.ipynb`
        slide_img (numpy.ndarray):
            the image to overlay
        mapping_path (str):
            the path to the file to save the mapping to
        draw_radius (int):
            the radius (in optical pixels) to draw each circle on the slide
        figsize (tuple):
            the size of the interactive figure to display
        check_dist (float):
            if the distance (in microns) between a manual-auto FOV pair exceeds this value, it will
            be reported for a potential error, if `None` distance will not be validated
        check_duplicates (bool):
            if `True`, validate whether an auto FOV has 2 manual FOVs mapping to it
        check_mismatches (bool):
            if `True`, validate whether the the manual auto FOV pairs have matching names
    """

    # error check: ensure mapping path exists
    if not os.path.exists(os.path.split(mapping_path)[0]):
        raise FileNotFoundError(
            "Path %s to mapping_path does not exist, "
            "please rename to a valid location" % os.path.split(mapping_path)[0]
        )

    # verify check_dist is positive if set as a numeric value
    dist_is_num = isinstance(check_dist, int) or isinstance(check_dist, float)
    if check_dist is not None and (not dist_is_num or check_dist <= 0):
        raise ValueError(
            "If validating distance, check_dist must be a positive floating point value"
        )

    # verify check_duplicates is a bool
    if not isinstance(check_duplicates, bool):
        raise ValueError("check_duplicates needs to be set to True or False")

    # verify check_mismatches is a bool
    if not isinstance(check_mismatches, bool):
        raise ValueError("check_mismatches needs to be set to True or False")

    # if there isn't a coreg_path defined, the user needs to run update_coregistration_params first
    if not os.path.exists(os.path.join('..', 'toffy', 'coreg_params.json')):
        raise FileNotFoundError(
            "You haven't co-registered your slide yet. Please run "
            "update_coregistraion_params.ipynb first."
        )

    # load the co-registration parameters in
    # NOTE: the last set of params in the coreg_params list is the most up-to-date
    with open(os.path.join('..', 'toffy', 'coreg_params.json')) as cp:
        stage_optical_coreg_params = json.load(cp)['coreg_params'][-1]

    # define the initial mapping and a distance lookup table between manual and auto FOVs
    manual_to_auto_map, manual_auto_dist = assign_closest_fovs(manual_fovs, auto_fovs)

    # condense manual_fovs to include just the name mapped to its coordinate
    # NOTE: convert to optical pixels for visualization
    manual_fovs_info = {
        fov['name']: convert_stage_to_optical(
            tuple(fov['centerPointMicrons'].values()), stage_optical_coreg_params
        )

        for fov in manual_fovs['fovs']
    }

    # sort manual FOVs by row then column, first assume the user names FOVs in R{m}c{n} format
    try:
        manual_fovs_sorted = sorted(
            list(manual_fovs_info.keys()),
            key=lambda mf: (int(re.findall(r'\d+', mf)[0]), int(re.findall(r'\d+', mf)[1]))
        )
    # otherwise, just sort manual FOVs alphabetically, nothing else we can do
    # NOTE: this will not catch cases where the user has something like fov2_data0
    except IndexError:
        warnings.warn(
            'Manual FOVs not consistently named in R{m}C{n} format, sorting alphabetically'
        )
        manual_fovs_sorted = sorted(list(manual_fovs_info.keys()))

    # get the first FOV to display
    first_manual = manual_fovs_sorted[0]

    # define the drop down menu for the manual fovs
    w_man = widgets.Dropdown(
        options=manual_fovs_sorted,
        value=first_manual,
        description='Manually-defined FOV',
        layout=widgets.Layout(width='auto'),
        style={'description_width': 'initial'}
    )

    # NOTE: convert to optical pixels for visualization
    auto_fovs_info = {
        fov: convert_stage_to_optical(auto_fovs[fov], stage_optical_coreg_params)

        for fov in auto_fovs
    }

    # sort FOVs alphabetically
    auto_fovs_sorted = sorted(
        list(auto_fovs.keys()),
        key=lambda af: (int(re.findall(r'\d+', af)[0]), int(re.findall(r'\d+', af)[1]))
    )

    # define the drop down menu for the auto fovs
    # the default value should be set to the auto fov the initial manual fov maps to
    w_auto = widgets.Dropdown(
        options=auto_fovs_sorted,
        value=manual_to_auto_map[first_manual],
        description='Automatically-generated FOV',
        layout=widgets.Layout(width='auto'),
        style={'description_width': 'initial'}
    )

    # define the save button
    w_save = widgets.Button(
        description='Save mapping',
        layout=widgets.Layout(width='auto'),
        style={'description_width': 'initial'}
    )

    # define the textbox to display the error message
    w_err = widgets.Textarea(
        description='FOV pair validation checks:',
        layout=widgets.Layout(height="auto", width='50%'),
        style={'description_width': 'initial'}
    )

    def increase_textarea_size(args):
        """Ensures size of `w_err` adjusts based on the amount of validation text needed

        Args:
            args (dict):
                the handler for `w_err`,
                only passed as a standard for `ipywidgets.Textarea` observe
        """
        w_err.rows = w_err.value.count('\n') + 1

    # ensure the entire error message is displayed
    w_err.observe(increase_textarea_size, 'value')

    # define a box to hold w_man and w_auto
    w_box = widgets.HBox(
        [w_man, w_auto, w_save],
        layout=widgets.Layout(
            display='flex',
            flex_flow='row',
            align_items='stretch',
            width='75%'
        )
    )

    # display the box with w_man and w_auto dropdown menus
    display(w_box)

    # display the w_err text box with validation errors
    display(w_err)

    # define an output context to display
    out = widgets.Output()

    # display the figure to plot on
    fig, ax = plt.subplots(figsize=figsize)

    # generate the circles and annotations for each circle to display on the image
    slide_img = generate_fov_circles(
        manual_fovs_info, auto_fovs_info, w_man.value, w_auto.value,
        slide_img, draw_radius
    )

    # make sure the output gets displayed to the output widget so it displays properly
    with out:
        # draw the image
        img_plot = ax.imshow(slide_img)

        # overwrite the default title
        _ = plt.title('Manually-defined to automatically-generated FOV map')

        # remove massive padding
        _ = plt.tight_layout()

        # define status of the save annotation, initially None, updates when user clicks w_save
        # NOTE: ipywidget callback functions can only access dicts defined in scope
        save_ann = {'annotation': None}

        # generate the annotation defining FOV pairings that need extra inspection
        # the user can specify as many of the following:
        # 1. the distance between the manual and auto FOV exceeds check_dist
        # 2. two manual FOVs map to the same auto FOV
        # 3. a manual FOV name does not match its auto FOV name
        manual_auto_warning = generate_validation_annot(
            manual_to_auto_map, manual_auto_dist, check_dist, check_duplicates, check_mismatches
        )

        # display the errors in the text box
        w_err.value = manual_auto_warning

    # a callback function for changing w_auto to the value w_man maps to
    # NOTE: needs to be here so it can access w_man and w_auto in scope
    def update_mapping(change):
        """Updates `w_auto` and bolds a different manual-auto pair when `w_prop` changes

        Args:
            change (dict):
                defines the properties of the changed value in `w_prop`
        """

        # only operate if w_prop actually changed
        # prevents update if the user drops down w_prop but leaves it as the same manual fov
        if change['name'] == 'value' and change['new'] != change['old']:
            # need to be in the output widget context to update
            with out:
                new_slide_img = update_mapping_display(
                    change, w_auto, manual_to_auto_map, manual_fovs_info, auto_fovs_info,
                    slide_img, draw_radius
                )

                # set the new slide img in the plot
                img_plot.set_data(new_slide_img)
                fig.canvas.draw_idle()

    # a callback function for remapping when w_auto changes
    # NOTE: needs to be here so it can access w_man and w_auto in scope
    def remap_values(change):
        """Bolds the new `w_auto` and maps the selected FOV in `w_man`
        to the new `w_auto` in `manual_to_auto_amp`

        Args:
            change (dict):
                defines the properties of the changed value in `w_auto`
        """

        # only remap if the auto change as been updated
        # prevents update if the user drops down w_auto but leaves it as the same auto fov
        if change['name'] == 'value' and change['new'] != change['old']:
            # need to be in the output widget context to update
            with out:
                new_slide_img, manual_auto_warning = remap_manual_to_auto_display(
                    change, w_man, manual_to_auto_map, manual_auto_dist,
                    auto_fovs_info, slide_img, draw_radius,
                    check_dist, check_duplicates, check_mismatches
                )

                # set the new slide img in the plot
                img_plot.set_data(new_slide_img)
                fig.canvas.draw_idle()

                # re-generate the validation warning in w_err
                w_err.value = manual_auto_warning

    # a callback function for saving manual_to_auto_map to mapping_path if w_save clicked
    def save_mapping(b):
        """Saves the mapping defined in `manual_to_auto_map`

        Args:
            b (ipywidgets.widgets.widget_button.Button):
                the button handler for `w_save`, only passed as a standard for `on_click` callback
        """

        # need to be in the output widget context to display status
        with out:
            # call the helper function to save manual_to_auto_map and notify user
            write_manual_to_auto_map(
                manual_to_auto_map, save_ann, mapping_path
            )

    # ensure a change to w_man redraws the image due to a new manual fov selected
    w_man.observe(update_mapping)

    # ensure a change to w_auto redraws the image due to a new automatic fov
    # mapped to the manual fov
    w_auto.observe(remap_values)

    # if w_save clicked, save the new mapping to the path defined in mapping_path
    w_save.on_click(save_mapping)

    # display the output
    display(out)


def remap_and_reorder_fovs(manual_fov_regions, manual_to_auto_map,
                           moly_path, randomize=False,
                           moly_insert=False, moly_interval=5):
    """Runs 3 separate tasks on `manual_fov_regions`:

    - Uses `manual_to_auto_map` to rename the FOVs
    - Randomizes the order of the FOVs (if specified)
    - Inserts Moly points at the specified interval (if specified)

    Args:
        manual_fov_regions (dict):
            The list of FOVs proposed by the user
        manual_to_auto_map (dict):
            Defines the mapping of manual to auto FOV names
        moly_path (str):
            The path to the Moly point to insert
        randomize (bool):
            Whether to randomize the FOVs
        moly_insert (bool):
            Whether to insert Moly points between FOVs at a specified `moly_interval`
        moly_interval (int):
            The interval in which to insert Moly points.
            Ignored if `moly_insert` is `False`.

    Returns:
        dict:
            `manual_fov_regions` with new FOV names, randomized, and with Moly points
    """

    # file path validation
    if not os.path.exists(moly_path):
        raise FileNotFoundError("Moly point %s does not exist" % moly_path)

    # load the Moly point in
    with open(moly_path, 'r', encoding='utf-8') as mp:
        moly_point = json.load(mp)

    # error check: moly_interval cannot be less than or equal to 0 if moly_insert is True
    if moly_insert and (not isinstance(moly_interval, int) or moly_interval < 1):
        raise ValueError("moly_interval must be a positive integer")

    # define a new fov regions dict for remapped names
    remapped_fov_regions = {}

    # copy over the metadata values from manual_fov_regions to remapped_fov_regions
    remapped_fov_regions = assign_metadata_vals(manual_fov_regions, remapped_fov_regions, ['fovs'])

    # define a new FOVs list for fov_regions_remapped
    remapped_fov_regions['fovs'] = []

    # rename the FOVs based on the mapping and append to fov_regions_remapped
    for fov in manual_fov_regions['fovs']:
        # needed to prevent early saving since interactive visualization cannot stop this
        # from running if a mapping_path provided already exists
        fov_data = copy.deepcopy(fov)

        # remap the name and append to fov_regions_remapped
        fov_data['name'] = manual_to_auto_map[fov['name']]
        remapped_fov_regions['fovs'].append(fov_data)

    # randomize the order of the FOVs if specified
    if randomize:
        remapped_fov_regions['fovs'] = shuffle(remapped_fov_regions['fovs'])

    # insert Moly points at the specified interval if specified
    if moly_insert:
        mi = moly_interval

        while mi < len(remapped_fov_regions['fovs']):
            remapped_fov_regions['fovs'].insert(mi, moly_point)
            mi += moly_interval + 1

    return remapped_fov_regions
