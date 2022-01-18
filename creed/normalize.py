# adapted from https://machinelearningmastery.com/curve-fitting-with-python/

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd




pulse_height = pd.read_csv('/Users/noahgreenwald/Downloads/trigger_1300_MPH.csv')
pulse_heights_final = pulse_height['median_intensity'].values[2:-5]

moly_counts = pd.read_csv('/Users/noahgreenwald/Downloads/trigger_1300_MPI_full_window.csv')
moly_counts_final = moly_counts['Mo98'].values[2:-5]


def create_objective_function(obj_func):
    valid_funcs = ['poly_3', 'poly_4', 'poly_5', 'log']
    if obj_func not in valid_funcs:
        raise ValueError('Invalid function, must be one of {}'.format(valid_funcs))
    if obj_func == 'poly_3':
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


# TODO: Will this be different for low- and high-mass channels?
def fit_calibration_curve(x_vals, y_vals, obj_func):

    # get objective function
    objective = create_objective_function(obj_func)

    # get fitted values
    popt, pcov = curve_fit(objective, x_vals, y_vals)

    # plot overlay of predicted fit and real values
    plt.scatter(x_vals, y_vals)
    x_line = np.arange(min(x_vals), max(x_vals), 1)
    y_line = objective(x_line, *popt)
    plt.plot(x_line, y_line, '--', color='red')

    # create prediction function based on fits

    return obj_func, popt


def normalize_image_data(data_dir, output_dir, fovs, pulse_heights, norm_value, calibration_func):

    # get FOVs to loop over
    if fovs is None:
        fovs = io_utils.list_folders(data_dir)

    pred_weights, pred_name = calibration_func
    obj_func = create_objective_function(pred_name)

    def pred_func(x):
        pred_value = obj_func(x, *pred_weights)
        return pred_value

    for fov in fovs:
        current_fov_dir = os.path.join(data_dir, fov)
        output_fov_dir = os.path.join(output_dir, fov)
        os.makedirs(output_fov_dir)

        images = load_utils.load_imgs_from_dir(current_fov_dir)

        fov_pulse_height = pulse_heights.loc[fov, 'pulse_height'].values

        fov_norm_factor = norm_value / fov_pulse_height

        normalized_images = images.values * fov_norm_factor


new_pred_func = fit_calibration_curve(pulse_heights=pulse_heights_final,
                                      signal_counts=moly_counts_final,
                                      obj_func='log')

new_pred_func(2500)