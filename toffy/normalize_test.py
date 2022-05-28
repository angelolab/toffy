import json
import numpy as np
import os
import pandas as pd
import pytest
import tempfile
import xarray as xr

from pytest_cases import parametrize_with_cases

from ark.utils import test_utils, load_utils
from toffy import normalize
import toffy.normalize_test_cases as test_cases

parametrize = pytest.mark.parametrize


def mocked_extract_bin_file(data_dir, include_fovs, panel, out_dir, intensities):
    mass_num = len(panel)

    base_img = np.ones((3, 4, 4))

    all_imgs = []
    for i in range(1, mass_num + 1):
        all_imgs.append(base_img * i)

    out_img = np.stack(all_imgs, axis=-1)

    out_img = np.expand_dims(out_img, axis=0)

    out_array = xr.DataArray(data=out_img,
                             coords=[
                                [include_fovs[0]],
                                ['pulse', 'intensity', 'area'],
                                np.arange(base_img.shape[1]),
                                np.arange(base_img.shape[2]),
                                panel['Target'].values,
                             ],
                             dims=['fov', 'type', 'x', 'y', 'channel'])
    return out_array


def mocked_pulse_height(data_dir, fov, panel, channel):
    mass = panel['Mass'].values[0]
    return mass * 2


def test_write_counts_per_mass(mocker):
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = os.path.join(temp_dir, 'out_dir')
        os.makedirs(out_dir)
        masses = [88, 89, 90]
        expected_counts = [16 * i for i in range(1, len(masses) + 1)]
        mocker.patch('toffy.normalize.extract_bin_files', mocked_extract_bin_file)

        normalize.write_counts_per_mass(base_dir=temp_dir, output_dir=out_dir, fov='fov1',
                                        masses=masses)
        output = pd.read_csv(os.path.join(out_dir, 'fov1_channel_counts.csv'))
        assert len(output) == len(masses)
        assert set(output['mass'].values) == set(masses)
        assert set(output['channel_count'].values) == set(expected_counts)


def test_write_mph_per_mass(mocker):
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = os.path.join(temp_dir, 'out_dir')
        os.makedirs(out_dir)
        masses = [88, 89, 90]
        mocker.patch('toffy.normalize.get_median_pulse_height', mocked_pulse_height)

        normalize.write_mph_per_mass(base_dir=temp_dir, output_dir=out_dir, fov='fov1',
                                     masses=masses)
        output = pd.read_csv(os.path.join(out_dir, 'fov1_pulse_heights.csv'))
        assert len(output) == len(masses)
        assert set(output['mass'].values) == set(masses)
        assert np.all(output['pulse_height'].values == output['mass'].values * 2)


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
            mph_dfs[i].to_csv(os.path.join(full_path, 'fov-1-scan-1_pulse_heights.csv'),
                              index=False)
            all_mph.extend(mph_dfs[i]['pulse_height'])

            count_dfs[i].to_csv(os.path.join(full_path, 'fov-1-scan-1_channel_counts.csv'),
                                index=False)
            all_counts.extend(count_dfs[i]['channel_count'])

            dir_paths.append(os.path.join(temp_dir, dir_names[i]))

        combined = normalize.combine_tuning_curve_metrics(dir_paths)

        # data may be in a different order due to matching dfs, but all values should be present
        assert set(all_mph) == set(combined['pulse_height'])
        assert set(all_counts) == set(combined['channel_count'])
        saved_dir_names = [name.split('/')[-1] for name in np.unique(combined['directory'])]
        assert set(saved_dir_names) == set(dir_names)

        # check that normalized value is 1 for maximum in each channel
        for mass in np.unique(combined['mass']):
            subset = combined.loc[combined['mass'] == mass, :]
            max = np.max(subset[['channel_count']].values)
            norm_vals = subset.loc[subset['channel_count'] == max, 'norm_channel_count'].values
            assert np.all(norm_vals == 1)


def test_normalize_image_data():
    with tempfile.TemporaryDirectory() as top_level_dir:
        data_dir = os.path.join(top_level_dir, 'data_dir')
        os.makedirs(data_dir)

        output_dir = os.path.join(top_level_dir, 'output_dir')
        os.makedirs(output_dir)

        # make fake data for testing
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=1, num_chans=10)
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            data_dir, fovs, chans, img_shape=(10, 10), fills=True)

        # weights of mph to norm const func: 0.01x + 0x^2 + 0.5
        weights = [0.01, 0, 0.5]
        name = 'poly_2'
        func_json = {'name': name, 'weights': weights}
        func_path = os.path.join(top_level_dir, 'norm_func.json')

        with open(func_path, 'w') as fp:
            json.dump(func_json, fp)

        # create pulse heights file with linearly increasing values
        masses = np.array(range(1, len(chans) + 1))
        panel_info_file = pd.DataFrame({'Mass': masses, 'Target': chans})
        mph_vals = np.arange(1, len(masses) + 1)

        pulse_heights = pd.DataFrame({'mass': masses,
                                     'fov': ['fov0' for _ in masses],
                                      'pulse_height': mph_vals})

        # normalize images
        normalize.normalize_image_data(data_dir, output_dir, fov='fov0',
                                       pulse_heights=pulse_heights, panel_info=panel_info_file,
                                       norm_func_path=func_path)

        normalized = load_utils.load_imgs_from_tree(output_dir, fovs=['fov0'], channels=chans)
        log_file = pd.read_csv(os.path.join(output_dir, 'fov0', 'normalization_coefs.csv'))

        # compute expected multipliers for each mass
        mults = mph_vals * weights[0] + weights[2]

        # check that image data has been rescaled appropriately
        mults = mults.reshape(1, 1, len(mults))
        assert np.allclose(data_xr.values, normalized.values * mults)

        # check that log file accurately recorded mults
        assert np.allclose(log_file['norm_vals'].values, mults)

        # check that warning is raised for out of range channels
        mph_vals[-1] = 100
        mph_vals[0] = -5
        pulse_heights = pd.DataFrame({'mass': masses,
                                      'fov': ['fov0' for _ in masses],
                                      'pulse_height': mph_vals})
        with pytest.warns(UserWarning, match='inspection for accuracy is recommended'):
            normalize.normalize_image_data(data_dir, output_dir, fov='fov0',
                                           pulse_heights=pulse_heights, panel_info=panel_info_file,
                                           norm_func_path=func_path)

        # bad function path
        with pytest.raises(ValueError, match='No normalization function'):
            normalize.normalize_image_data(data_dir, output_dir, fov='fov0',
                                           pulse_heights=pulse_heights, panel_info=panel_info_file,
                                           norm_func_path='bad_func_path')