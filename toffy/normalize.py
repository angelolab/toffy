# adapted from https://machinelearningmastery.com/curve-fitting-with-python/

import json
import os
import shutil
import warnings

import numpy as np
from scipy.optimize import curve_fit
import skimage.io as io
import matplotlib.pyplot as plt
import pandas as pd

from ark.utils import io_utils, load_utils
from mibi_bin_tools.bin_files import extract_bin_files, get_median_pulse_height
from mibi_bin_tools.panel_utils import make_panel


def write_counts_per_mass(base_dir, output_dir, fov, masses, integration_window=(0.5, 0.5)):
    """Records the total counts per mass for the specified FOV

    Args:
        base_dir (str): the directory containing the FOV
        output_dir (str): the directory where the csv file will be saved
        fov (str): the name of the fov to extract
        masses (list): the list of masses to extract counts from
        integration_window (tuple): start and stop offset for integrating mass peak
    """

    start_offset, stop_offset = integration_window

    # create panel with extraction criteria
    panel = make_panel(mass=masses, low_range=start_offset, high_range=stop_offset)

    array = extract_bin_files(data_dir=base_dir, out_dir=None, include_fovs=[fov], panel=panel,
                              intensities=False)
    # we only care about pulse counts, not intensities
    array = array.loc[fov, 'pulse', :, :, :]
    channel_count = np.sum(array, axis=(0, 1))

    # create df to hold output
    fovs = np.repeat(fov, len(masses))
    out_df = pd.DataFrame({'mass': masses,
                           'fov': fovs,
                           'channel_count': channel_count})
    out_df.to_csv(os.path.join(output_dir, fov + '_channel_counts.csv'), index=False)


def write_mph_per_mass(base_dir, output_dir, fov, masses, integration_window=(0.5, 0.5)):
    """Records the mean pulse height (MPH) per mass for the specified FOV

    Args:
        base_dir (str): the directory containing the FOV
        output_dir (str): the directory where the csv file will be saved
        fov (str): the name of the fov to extract
        masses (list): the list of masses to extract MPH from
        integration_window (tuple): start and stop offset for integrating mass peak
    """
    start_offset, stop_offset = integration_window

    # hold computed values
    mph_vals = []

    for mass in masses:
        panel = make_panel(mass=mass, low_range=start_offset, high_range=stop_offset)
        mph_vals.append(get_median_pulse_height(data_dir=base_dir, fov=fov, channel='targ0',
                                                panel=panel))
    # create df to hold output
    fovs = np.repeat(fov, len(masses))
    out_df = pd.DataFrame({'mass': masses,
                           'fov': fovs,
                           'pulse_height': mph_vals})
    out_df.to_csv(os.path.join(output_dir, fov + '_pulse_heights.csv'), index=False)


def create_objective_function(obj_func):
    """Creates a function of specified type to be used for fitting a curve

    Args:
        obj_func (str): the desired objective function. Must be either poly_2, ..., poly_5, or log

    Returns:
        function: the function which will be optimized"""

    # input validation
    valid_funcs = ['poly_2', 'poly_3', 'poly_4', 'poly_5', 'log', 'exp']
    if obj_func not in valid_funcs:
        raise ValueError('Invalid function, must be one of {}'.format(valid_funcs))

    # define objective functions
    def poly_2(x, a, b, c):
        return a * x + b * x ** 2 + c

    def poly_3(x, a, b, c, d):
        return a * x + b * x ** 2 + c * x ** 3 + d

    def poly_4(x, a, b, c, d, e):
        return a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e

    def poly_5(x, a, b, c, d, e, f):
        return a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e * x ** 5 + f

    def log(x, a, b):
        return a * np.log(x) + b

    def exp(x, a, b, c, d):
        x_log = np.log(x)
        return a * x_log + b * x_log ** 2 + c * x_log ** 3 + d

    objectives = {'poly_2': poly_2, 'poly_3': poly_3, 'poly_4': poly_4, 'poly_5': poly_5,
                  'log': log, 'exp': exp}

    return objectives[obj_func]


def fit_calibration_curve(x_vals, y_vals, obj_func, plot_fit=False):
    """Finds the optimal weights to fit the supplied values for the specified function

    Args:
        x_vals (list): the x values to be fit
        y_vals (list): the y value to be fit
        obj_func (str): the name of the function that will be fit to the data
        plot_fit (bool): whether or not to plot the fit of the function vs the values

    Returns:
        list: the weights of the fitted function"""

    # get objective function
    objective = create_objective_function(obj_func)

    # get fitted values
    popt, _ = curve_fit(objective, x_vals, y_vals)

    # plot overlay of predicted fit and real values
    if plot_fit:
        plt.scatter(x_vals, y_vals)
        x_line = np.arange(min(x_vals), max(x_vals), 1)
        y_line = objective(x_line, *popt)
        plt.plot(x_line, y_line, '--', color='red')

    return popt


def create_prediction_function(name, weights):
    """Creates a prediction function given a specified function type and fitted weights

    Args:
        name (str): name of the function to use
        weights (list): list of fitted weights

    Returns:
        func: prediction function"""

    # get function based on specified pred type
    obj_func = create_objective_function(name)

    # define function which takes only x as input, uses weights for remaining parameters
    def pred_func(x):
        output = obj_func(x, *weights)
        return output

    return pred_func


def combine_run_metrics(run_dir, file_prefix):
    """Combines the specified metrics from different FOVs into a single file

    Args:
        run_dir (str): the directory containing the files
        file_prefix (str): the prefix of the files to be combined"""

    files = io_utils.list_files(run_dir, file_prefix)
    bins = io_utils.list_files(run_dir, '.bin')

    # validate inputs
    if len(bins) == 0:
        raise ValueError('No bin files found in {}'.format(run_dir))

    if file_prefix + '_combined.csv' in files:
        warnings.warn('removing previously generated '
                      'combined {} file in {}'.format(file_prefix, run_dir))
        os.remove(os.path.join(run_dir, file_prefix + '_combined.csv'))
        files = [file for file in files if 'combined' not in file]

    if len(bins) != len(files):
        raise ValueError('Mismatch between the number of bins and number '
                         'of {} files in {}'.format(file_prefix, run_dir))

    # collect all metrics files
    metrics = []
    for file in files:
        metrics.append(pd.read_csv(os.path.join(run_dir, file)))

    # check that all files are the same length
    if len(metrics) > 1:
        base_len = len(metrics[0])
        for i in range(1, len(metrics)):
            if len(metrics[i]) != base_len:
                raise ValueError('Not all {} files are the same length: file {} does not match'
                                 'file {}'.format(file_prefix, files[0], files[i]))

    metrics = pd.concat(metrics)

    metrics.to_csv(os.path.join(run_dir, file_prefix + '_combined.csv'), index=False)


def combine_tuning_curve_metrics(dir_list):
    """Combines metrics together into a single dataframe for fitting a turning curve

    Args:
        dir_list (list): list of directories to pull metrics from

    Returns:
        pd.DataFrame: dataframe containing aggregates metrics"""

    # create list to hold all extracted data
    all_dirs = []

    # loop through each run folder
    for dir in dir_list:

        # combine tables together
        pulse_heights = pd.read_csv(os.path.join(dir, 'fov-1-scan-1_pulse_heights.csv'))
        channel_counts = pd.read_csv(os.path.join(dir, 'fov-1-scan-1_channel_counts.csv'))
        combined = pulse_heights.merge(channel_counts, 'outer', on=['fov', 'mass'])

        if len(combined) != len(pulse_heights):
            raise ValueError("Pulse heights and channel counts must be generated for the same "
                             "mass ranges and fovs. However, the data the following does do not "
                             "exactly match: {}".format(dir))

        # add directory label and add to list
        combined['directory'] = dir
        all_dirs.append(combined)

    # combine data from each dir together
    all_data = pd.concat(all_dirs)
    all_data.reset_index()

    # create normalized counts column
    subset = all_data[['channel_count', 'mass']]
    all_data['norm_channel_count'] = subset.groupby('mass').transform(lambda x: (x / x.max()))

    return all_data


def normalize_image_data(img_dir, output_dir, fov, pulse_heights, panel_info, img_sub_folder='',
                         norm_func_path=os.path.join('..', 'toffy', 'norm_func.json'),
                         extreme_vals=(0.5, 1)):
    """Normalizes image data based on median pulse height from the run and a tuning curve

    Args:
        img_dir (str): directory with the image data
        output_dir (str): directory where the normalized images will be saved
        fov (str): name of the fov to normalize
        pulse_heights (pd.DataFrame): pulse heights per mass
        panel_info (pd.DataFrame): mapping between channels and masses
        norm_func_path (str): file containing the saved weights for the normalization function
        extreme_vals (tuple): determines the range for norm vals which will raise a warning
    """
    # load normalization function
    if not os.path.exists(norm_func_path):
        raise ValueError("No normalization function found. You will need to run "
                         "section 3 of the 1_set_up_toffy.ipynb notebook to generate the "
                         "necessary function before you can normalize your data")

    with open(norm_func_path, 'r') as cf:
        norm_json = json.load(cf)

    norm_weights, norm_name = norm_json['weights'], norm_json['name']

    channels = panel_info['Target'].values

    # instantiate function which translates pulse height to a normalization constant
    norm_func = create_prediction_function(norm_name, norm_weights)

    output_fov_dir = os.path.join(output_dir, fov)
    if os.path.exists(output_fov_dir):
        print("output directory {} already exists, "
              "data will be overwritten".format(output_fov_dir))
    else:
        os.makedirs(output_fov_dir)

    # get images and pulse heights
    images = load_utils.load_imgs_from_tree(img_dir, fovs=[fov], channels=channels,
                                            dtype='float32', img_sub_folder=img_sub_folder)

    # predict normalization based on MPH value for all masses
    norm_vals = norm_func(pulse_heights['pulse_height'].values)

    # check if any values are outside expected range
    extreme_mask = np.logical_or(norm_vals < extreme_vals[0], norm_vals > extreme_vals[1])
    if np.any(extreme_mask):
        bad_channels = channels[extreme_mask]
        warnings.warn('The following channel(s) had an extreme normalization value. Manual '
                      'inspection for accuracy is recommended: {}'.format(bad_channels))

    # correct images and save
    normalized_images = images / norm_vals.reshape((1, 1, 1, len(channels)))

    for idx, chan in enumerate(channels):
        io.imsave(os.path.join(output_fov_dir, chan + '.tiff'),
                  normalized_images[0, :, :, idx], check_contrast=False)

    log_df = pd.DataFrame({'channels': channels,
                           'norm_vals': norm_vals})
    log_df.to_csv(os.path.join(output_fov_dir, 'normalization_coefs.csv'), index=False)