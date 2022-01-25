# adapted from https://machinelearningmastery.com/curve-fitting-with-python/

import json
import os
import warnings

import numpy as np
from scipy.optimize import curve_fit
import skimage.io as io
import matplotlib.pyplot as plt
import pandas as pd


def create_objective_function(obj_func):
    """Creates a function of specified type to be used for fitting a curve

    Args:
        obj_func (str): the desired objective function. Must be either poly_2, ..., poly_5, or log

    Returns:
        function: the function which will be optimized"""

    valid_funcs = ['poly_2', 'poly_3', 'poly_4', 'poly_5', 'log']
    if obj_func not in valid_funcs:
        raise ValueError('Invalid function, must be one of {}'.format(valid_funcs))

    if obj_func == 'poly_2':
        def objective(x, a, b, c):
            return a * x + b * x ** 2 + c
    elif obj_func == 'poly_3':
        def objective(x, a, b, c, d):
            return a * x + b * x ** 2 + c * x ** 3 + d
    elif obj_func == 'poly_4':
        def objective(x, a, b, c, d, e):
            return a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e
    elif obj_func == 'poly_5':
        def objective(x, a, b, c, d, e, f):
            return a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e * x ** 5 + f
    else:
        def objective(x, a, b):
            return a * np.log(x) + b

    return objective


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
    popt, pcov = curve_fit(objective, x_vals, y_vals)

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


def combine_fov_metrics(dir_list, num_fovs):
    """Combines metrics for data normalization together into a single dataframe

    Args:
        dir_list (list): list of directories
        num_fovs (int): number of fovs present within each directory

    Returns:
        pd.DataFrame: dataframe containing aggregates metrics"""

    # create list to hold all extracted data
    all_dirs = []

    # loop through directories, and if present, multiple fovs within directories
    for dir in dir_list:
        all_fovs = []
        for fov in range(1, num_fovs + 1):
            pulse_heights = pd.read_csv(os.path.join(dir, 'pulse_heights_{}.csv'.format(fov)))
            channel_counts = pd.read_csv(os.path.join(dir, 'channel_counts_{}.csv'.format(fov)))

            if not np.all(pulse_heights['masses'] == channel_counts['masses']):
                raise ValueError("Pulse counts and channel counts must be generated for the same"
                                 "mass range. However, the following data contain different"
                                 "masses: directory {}, fov {}".format(dir, fov))

            # combine into single df per fov, and add to list for entire directory
            pulse_heights['channel_counts'] = channel_counts['counts']
            pulse_heights['fov'] = fov
            all_fovs.append(pulse_heights)

        # combine data from all fovs into single df, and add to list for all directories
        fov_df = pd.concat(all_fovs)
        fov_df['directory'] = dir
        all_dirs.append(fov_df)

    # combine data from each dir together
    all_data = pd.concat(all_dirs)
    all_data.reset_index()

    # create normalized counts column
    subset = all_data[['channel_counts', 'masses']]
    all_data['norm_channel_counts'] = subset.groupby('masses').transform(lambda x: (x / x.max()))

    return all_data


def normalize_image_data(data_dir, output_dir, fovs, pulse_heights, panel_info_path,
                         calibration_func_path):
    """Normalizes image data based on median pulse height

    """

    # get FOVs to loop over
    if fovs is None:
        fovs = io_utils.list_folders(data_dir)

    # load calibration function
    with open(calibration_func_path, 'r') as cf:
        calibration_json = json.load(cf)

    cal_weights, cal_name = calibration_json['weights'], calibration_json['name']

    panel_info = pd.read_csv(panel_info_path)
    channels = panel_info['targets'].values

    # instantiate function which translates pulse height to a normalization constant
    calibration_func = create_prediction_function(cal_name, cal_weights)

    for fov in fovs:
        current_fov_dir = os.path.join(data_dir, fov)
        output_fov_dir = os.path.join(output_dir, fov)
        os.makedirs(output_fov_dir)

        # get images and pulse heights for current fov
        images = load_utils.load_imgs_from_dir(current_fov_dir, channels=channels)
        fov_pulse_heights = pulse_heights.loc[pulse_heights['fov'] == fov, :]

        # fit a function to model pulse height as a function of mass
        mass_weights = fit_calibration_curve(x=fov_pulse_heights['masses'],
                                             y=fov_pulse_heights['mphs'],
                                             obj_func='poly_2')
        mass_func = create_prediction_function(name='poly_2', weights=mass_weights)
        norm_vals = mass_func[panel_info['masses']].reshape((1, 1, 1, len(channels)))

        if np.any(norm_vals < 0.5 or norm_vals > 1.3):
            warnings.warn('The following FOV had an extreme normalization value. Manually '
                          'inspection for accuracy is recommended: fov {}'.format(fov))

        normalized_images = images / norm_vals

        for idx, chan in channels:
            io.imsave(os.path.join(output_fov_dir, chan + '.tiff'), normalized_images[0, :, :, idx])




