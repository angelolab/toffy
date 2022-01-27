import os
import pytest

import numpy as np
import pandas as pd
import tempfile


from toffy import normalize

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


def test_combine_run_metrics():
    with tempfile.TemporaryDirectory() as temp_dir:

        # create full directory of files
        bins = []
        metrics = []
        for i in range(1, 5):
            bin_name = 'example_{}.bin'.format(i)
            bins.append(bin_name)
            metric_name = 'example_metric_{}.csv'.format(i)
            metric_values = {'column_1': np.random.rand(10),
                             'column_2': np.random.rand(10),
                             'column_3': np.random.rand(10)}
            metrics.append([metric_name, metric_values])

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


def test_combine_tuning_curve_metrics():
    with tempfile.TemporaryDirectory() as temp_dir:

        dirs = ['dir_{}'.format(i) for i in range(1, 4)]
        dir_paths = [os.path.join(temp_dir, dir) for dir in dirs]

        for dir in dirs[1:3]:
            os.makedirs(os.path.join(temp_dir, dir))
            masses = np.random.randint(0, 10, 13)
            masses_rev = np.flip(masses)
            mph_vals = np.repeat(1, 13)
            channel_counts = np.arange(3, 16)
            fovs = np.arange(13)
            #fovs_rev = list(range(14, 1, -1))
            fovs_rev = np.flip(fovs)

            mph_df = pd.DataFrame({'masses': masses, 'fovs': fovs, 'mph': mph_vals})

            count_df = pd.DataFrame({'masses': masses_rev, 'channel_counts': channel_counts,
                                     'fovs': fovs_rev})

            mph_df.to_csv(os.path.join(temp_dir, dir, 'pulse_heights_combined.csv'), index=False)
            count_df.to_csv(os.path.join(temp_dir, dir, 'channel_counts_combined.csv'), index=False)

        combined = normalize.combine_tuning_curve_metrics(dir_paths[1:3])

        # data may be in a different order due to matching dfs, but all values should be present
        #for key in ['masses', 'channel_counts', 'fovs']:
        for key in ['channel_counts']:

            print(key)
            assert set(combined[key].values) == set(count_df[key].values)

        assert set(combined['mph'].values) == set(mph_df['mph'].values)

        # check that normalized values make sense
        # check that multiple folders work


