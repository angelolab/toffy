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
from toffy.json_utils import read_json_file


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
                         'of {} files'.format(file_prefix))

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

    # loop through directories, and if present, multiple fovs within directories
    for dir in dir_list:

        # generate aggregated table if it doesn't already exist
        for prefix in ['pulse_heights', 'channel_counts']:
            if not os.path.exists(os.path.join(dir, prefix + '_combined.csv')):
                combine_run_metrics(dir, prefix)

        # combine tables together
        pulse_heights = pd.read_csv(os.path.join(dir, 'pulse_heights_combined.csv'))
        channel_counts = pd.read_csv(os.path.join(dir, 'channel_counts_combined.csv'))
        combined = pulse_heights.merge(channel_counts, 'outer', on=['fovs', 'masses'])

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
    subset = all_data[['channel_counts', 'masses']]
    all_data['norm_channel_counts'] = subset.groupby('masses').transform(lambda x: (x / x.max()))

    return all_data


def normalize_image_data(data_dir, output_dir, fovs, pulse_heights, panel_info,
                         norm_func_path, mph_func_type='poly_2', extreme_vals=(0.5, 1)):
    """Normalizes image data based on median pulse height from the run and a tuning curve

    Args:
        data_dir (str): directory with the image data
        output_dir (str): directory where the normalized images will be saved
        fovs (list or None): which fovs to include in normalization. If None, uses all fovs
        pulse_heights (pd.DataFrame): pulse heights per mass per fov
        panel_info (pd.DataFrame): mapping between channels and masses
        norm_func_path (str): file containing the saved weights for the normalization function
        mph_func_type (str): name of the function to use for fitting the mass vs mph curve
        extreme_vals (tuple): determines the range for norm vals which will raise a warning
    """

    # get FOVs to loop over
    if fovs is None:
        fovs = io_utils.list_folders(data_dir)

    # load calibration function

    norm_json = read_json_file(norm_func_path)

    norm_weights, norm_name = norm_json['weights'], norm_json['name']

    channels = panel_info['targets'].values

    # instantiate function which translates pulse height to a normalization constant
    norm_func = create_prediction_function(norm_name, norm_weights)

    for fov in fovs:
        output_fov_dir = os.path.join(output_dir, fov)
        os.makedirs(output_fov_dir)

        # get images and pulse heights for current fov
        images = load_utils.load_imgs_from_tree(data_dir, fovs=[fov], channels=channels,
                                                dtype='float32')
        fov_pulse_heights = pulse_heights.loc[pulse_heights['fov'] == fov, :]

        # fit a function to model pulse height as a function of mass
        mph_weights = fit_calibration_curve(x_vals=fov_pulse_heights['masses'].values,
                                            y_vals=fov_pulse_heights['mphs'].values,
                                            obj_func=mph_func_type)

        # predict mph for each mass in the panel
        mph_func = create_prediction_function(name=mph_func_type, weights=mph_weights)
        mph_vals = mph_func(panel_info['masses'].values)

        # predict normalization for each mph in the panel
        norm_vals = norm_func(mph_vals)

        if np.any(norm_vals < extreme_vals[0]) or np.any(norm_vals > extreme_vals[1]):
            warnings.warn('The following FOV had an extreme normalization value. Manually '
                          'inspection for accuracy is recommended: fov {}'.format(fov))

        normalized_images = images / norm_vals.reshape((1, 1, 1, len(channels)))

        for idx, chan in enumerate(channels):
            io.imsave(os.path.join(output_fov_dir, chan + '.tiff'),
                      normalized_images[0, :, :, idx], check_contrast=False)
