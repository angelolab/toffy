import json
import os
import pytest

import numpy as np
import pandas as pd
import tempfile

from pytest_cases import parametrize_with_cases

from ark.utils import test_utils, load_utils
from toffy import normalize
import toffy.normalize_test_cases as test_cases

parametrize = pytest.mark.parametrize


# TODO: move to toolbox repo once created
def _make_blank_file(folder, name):
    with open(os.path.join(folder, name), 'w'):
        pass


@parametrize('obj_func_name, num_params', [('poly_2', 3), ('poly_3', 4), ('poly_4', 5),
                                           ('poly_5', 6), ('log', 2), ('exp', 4)])
def test_create_objective_function(obj_func_name, num_params):

    obj_func = normalize.create_objective_function(obj_func_name)

    # number of weights + 1 for x
    inputs = [1] * (num_params + 1)

    _ = obj_func(*inputs)


@parametrize('plot_fit', [True, False])
@parametrize('obj_func', ['poly_2', 'poly_3', 'poly_4', 'poly_5', 'log', 'exp'])
def test_fit_calibration_curve(plot_fit, obj_func):
    x_vals = np.random.rand(15)
    y_vals = np.random.rand(15)
    _ = normalize.fit_calibration_curve(x_vals, y_vals, obj_func, plot_fit)


@parametrize('obj_func, num_params', [('poly_2', 3), ('poly_3', 4), ('poly_4', 5),
                                      ('poly_5', 6), ('log', 2), ('exp', 4)])
def test_create_prediction_function(obj_func, num_params):
    weights = np.random.rand(num_params)
    pred_func = normalize.create_prediction_function(obj_func, weights)

    _ = pred_func(np.random.rand(10))


@parametrize_with_cases('bins, metrics', cases=test_cases.CombineRunMetricFiles)
def test_combine_run_metrics(bins, metrics):
    with tempfile.TemporaryDirectory() as temp_dir:

        for bin_file in bins:
            _make_blank_file(temp_dir, bin_file)

        for metric in metrics:
            name, values_df = metric[0], pd.DataFrame(metric[1])
            values_df.to_csv(os.path.join(temp_dir, name), index=False)

        normalize.combine_run_metrics(temp_dir, 'example_metric')

        combined_data = pd.read_csv(os.path.join(temp_dir, 'example_metric_combined.csv'))

        assert np.array_equal(combined_data.columns, ['column_1', 'column_2', 'column_3'])
        assert len(combined_data) == len(bins) * 10

        # check that previously generated combined file is removed with warning
        with pytest.warns(UserWarning, match='previously generated'):
            normalize.combine_run_metrics(temp_dir, 'example_metric')

        # check that files with different lengths raises error
        name, bad_vals = metrics[0][0], pd.DataFrame(metrics[0][1])
        bad_vals = bad_vals.loc[0:5, :]
        bad_vals.to_csv(os.path.join(temp_dir, name), index=False)

        with pytest.raises(ValueError, match='files are the same length'):
            normalize.combine_run_metrics(temp_dir, 'example_metric')
        os.remove(os.path.join(temp_dir, name))
        os.remove(os.path.join(temp_dir, 'example_1.bin'))

        # different number of bins raises error
        os.remove(os.path.join(temp_dir, bins[3]))
        with pytest.raises(ValueError, match='Mismatch'):
            normalize.combine_run_metrics(temp_dir, 'example_metric')

        # empty directory raises error
        empty_dir = os.path.join(temp_dir, 'empty')
        os.makedirs(empty_dir)

        with pytest.raises(ValueError, match='No bin files'):
            normalize.combine_run_metrics(empty_dir, 'example_metric')


@parametrize_with_cases('dir_names, mph_dfs, count_dfs', test_cases.TuningCurveFiles)
def test_combine_tuning_curve_metrics(dir_names, mph_dfs, count_dfs):
    with tempfile.TemporaryDirectory() as temp_dir:

        # variables to hold all unique values of each metric
        all_mph, all_counts, dir_paths = [], [], []

        # create csv files with data to be combined
        for i in range(len(dir_names)):
            full_path = os.path.join(temp_dir, dir_names[i])
            os.makedirs(full_path)
            mph_dfs[i].to_csv(os.path.join(full_path, 'pulse_heights_combined.csv'), index=False)
            all_mph.extend(mph_dfs[i]['mph'])

            count_dfs[i].to_csv(os.path.join(full_path, 'channel_counts_combined.csv'),
                                index=False)
            all_counts.extend(count_dfs[i]['channel_counts'])

            dir_paths.append(os.path.join(temp_dir, dir_names[i]))

        combined = normalize.combine_tuning_curve_metrics(dir_paths)

        # data may be in a different order due to matching dfs, but all values should be present
        assert set(all_mph) == set(combined['mph'])
        assert set(all_counts) == set(combined['channel_counts'])
        saved_dir_names = [name.split('/')[-1] for name in np.unique(combined['directory'])]
        assert set(saved_dir_names) == set(dir_names)

        # check that normalized value is 1 for maximum in each channel
        for mass in np.unique(combined['masses']):
            subset = combined.loc[combined['masses'] == mass, :]
            max = np.max(subset[['channel_counts']].values)
            norm_vals = subset.loc[subset['channel_counts'] == max, 'norm_channel_counts'].values
            assert np.all(norm_vals == 1)


def test_normalize_image_data():
    with tempfile.TemporaryDirectory() as top_level_dir:
        data_dir = os.path.join(top_level_dir, 'data_dir')
        os.makedirs(data_dir)

        output_dir = os.path.join(top_level_dir, 'output_dir')
        os.makedirs(output_dir)

        # make fake data for testing
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=2, num_chans=10)
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            data_dir, fovs, chans, img_shape=(10, 10), fills=True)

        # weights of mph to norm const func: 0.1x + 0x^2 + 0.5
        weights = [0.1, 0, 0.5]
        name = 'poly_2'
        func_json = {'name': name, 'weights': weights}
        func_path = os.path.join(top_level_dir, 'norm_func.json')

        with open(func_path, 'w') as fp:
            json.dump(func_json, fp)

        masses = np.array(range(1, len(chans) + 1))
        panel_info_file = pd.DataFrame({'masses': masses, 'targets': chans})

        # fov1 mass to mph function: line with 20% slope starting at 1.2
        mph_0 = masses * 0.2 + 1
        fov_0 = ['fov0'] * len(chans)

        # fov1 mass to mph function: line with 10% slope starting at 4
        mph_1 = masses * .1 + 4
        fov_1 = ['fov1'] * len(chans)

        pulse_heights = pd.DataFrame({'masses': np.concatenate([masses, masses]),
                                     'fov': fov_0 + fov_1,
                                      'mphs': np.concatenate([mph_0, mph_1])})

        normalize.normalize_image_data(data_dir, output_dir, fovs=None,
                                       pulse_heights=pulse_heights, panel_info=panel_info_file,
                                       norm_func_path=func_path)

        normalized = load_utils.load_imgs_from_tree(output_dir, fovs=fovs, channels=chans)

        # compute expected multipliers for each mass in each fov
        mults = [mph_array * weights[0] + weights[2] for mph_array in [mph_0, mph_1]]

        for idx, mult in enumerate(mults):
            mult = mult.reshape(1, 1, len(mult))
            assert np.allclose(data_xr.values[idx, :, :, :],
                               normalized.values[idx, :, :, :] * mult)
