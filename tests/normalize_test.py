import json
import os
import pathlib
import shutil
import tempfile
from typing import List
from unittest.mock import patch

import natsort
import natsort as ns
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from alpineer import io_utils, load_utils, test_utils
from pytest import TempPathFactory
from pytest_cases import parametrize_with_cases

from toffy import normalize
from toffy.json_utils import read_json_file, write_json_file

from .utils import normalize_test_cases as test_cases

parametrize = pytest.mark.parametrize


def mocked_extract_bin_file(data_dir, include_fovs, panel, out_dir, intensities):
    mass_num = len(panel)

    base_img = np.ones((3, 4, 4))

    all_imgs = []
    for i in range(1, mass_num + 1):
        all_imgs.append(base_img * i)

    out_img = np.stack(all_imgs, axis=-1)

    out_img = np.expand_dims(out_img, axis=0)

    out_array = xr.DataArray(
        data=out_img,
        coords=[
            [include_fovs[0]],
            ["pulse", "intensity", "area"],
            np.arange(base_img.shape[1]),
            np.arange(base_img.shape[2]),
            panel["Target"].values,
        ],
        dims=["fov", "type", "x", "y", "channel"],
    )
    return out_array


def mocked_pulse_height(data_dir, fov, panel, channels):
    return {chan: chan * 2 for chan in channels}


def test_write_counts_per_mass(mocker):
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = os.path.join(temp_dir, "out_dir")
        os.makedirs(out_dir)
        masses = [88, 89, 90]
        expected_counts = [16 * i for i in range(1, len(masses) + 1)]
        mocker.patch("toffy.normalize.extract_bin_files", mocked_extract_bin_file)

        normalize.write_counts_per_mass(
            base_dir=temp_dir, output_dir=out_dir, fov="fov1", masses=masses
        )
        output = pd.read_csv(os.path.join(out_dir, "fov1_channel_counts.csv"))
        assert len(output) == len(masses)
        assert set(output["mass"].values) == set(masses)
        assert set(output["channel_count"].values) == set(expected_counts)


def test_write_mph_per_mass(mocker):
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = os.path.join(temp_dir, "out_dir")
        os.makedirs(out_dir)
        masses = [88, 89, 90]
        mocker.patch("toffy.normalize.get_median_pulse_height", mocked_pulse_height)

        normalize.write_mph_per_mass(
            base_dir=temp_dir, output_dir=out_dir, fov="fov1", masses=masses
        )
        output = pd.read_csv(os.path.join(out_dir, "fov1_pulse_heights.csv"))
        assert len(output) == len(masses)
        assert set(output["mass"].values) == set(masses)
        assert np.all(output["pulse_height"].values == output["mass"].values * 2)

        normalize.write_mph_per_mass(
            base_dir=temp_dir, output_dir=out_dir, fov="fov1", masses=masses, proficient=True
        )
        assert os.path.exists(os.path.join(out_dir, "fov1_pulse_heights_proficient.csv"))


@parametrize(
    "obj_func_name, num_params",
    [
        ("poly_2", 3),
        ("poly_3", 4),
        ("poly_4", 5),
        ("poly_5", 6),
        ("log", 2),
        ("exp", 4),
    ],
)
def test_create_objective_function(obj_func_name, num_params):
    obj_func = normalize.create_objective_function(obj_func_name)

    # number of weights + 1 for x
    inputs = [1] * (num_params + 1)
    obj_func(*inputs)

    # test out with an invalid value for log functions
    inputs = [0] * (num_params + 1)
    obj_func(*inputs)

    # test with a list of valid values
    inputs = [np.array([1, 1, 1])]
    inputs.extend([1] * num_params)
    obj_func(*inputs)

    # test with a list of invalid values
    inputs = [np.array([0, 0, 0])]
    inputs.extend([1] * num_params)
    obj_func(*inputs)

    # test with a mix of valid and invalid values
    inputs = [np.array([0, 1, 1])]
    inputs.extend([1] * num_params)
    obj_func(*inputs)


@parametrize("plot_fit", [True, False])
@parametrize("obj_func", ["poly_2", "poly_3", "poly_4", "poly_5", "log", "exp"])
def test_fit_calibration_curve(plot_fit, obj_func):
    x_vals = np.random.rand(15)
    y_vals = np.random.rand(15)
    _ = normalize.fit_calibration_curve(x_vals, y_vals, obj_func, plot_fit)


@parametrize(
    "obj_func, num_params",
    [
        ("poly_2", 3),
        ("poly_3", 4),
        ("poly_4", 5),
        ("poly_5", 6),
        ("log", 2),
        ("exp", 4),
    ],
)
def test_create_prediction_function(obj_func, num_params):
    weights = np.random.rand(num_params)
    pred_func = normalize.create_prediction_function(obj_func, weights)

    _ = pred_func(np.random.rand(10))


@parametrize_with_cases("metrics", cases=test_cases.CombineRunMetricFiles)
@parametrize("warn_overwrite_test", [True, False])
def test_combine_run_metrics(metrics, warn_overwrite_test):
    with tempfile.TemporaryDirectory() as temp_dir:
        for metric in metrics["deficient"]:
            name, values_df = metric[0], pd.DataFrame(metric[1])
            values_df.to_csv(os.path.join(temp_dir, name), index=False)

        for metric in metrics["proficient"]:
            name_prof, values_df_prof = metric[0], pd.DataFrame(metric[1])
            values_df_prof.to_csv(os.path.join(temp_dir, name_prof), index=False)

        normalize.combine_run_metrics(temp_dir, "pulse_heights")

        combined_data = pd.read_csv(os.path.join(temp_dir, "pulse_heights_combined.csv"))

        assert np.array_equal(combined_data.columns, ["pulse_height", "mass", "fov"])
        assert len(combined_data) == len(metrics["deficient"]) * 10

        # check that previously generated combined file is removed with warning
        # NOTE: only if warn_overwrite turned on
        if warn_overwrite_test:
            with pytest.warns(UserWarning, match="previously generated"):
                normalize.combine_run_metrics(temp_dir, "pulse_heights", warn_overwrite_test)
        else:
            normalize.combine_run_metrics(temp_dir, "pulse_heights", warn_overwrite_test)

        # check that files with different lengths raises error
        name, bad_vals = metrics["deficient"][0][0], pd.DataFrame(metrics["deficient"][0][1])
        bad_vals = bad_vals.loc[0:5, :]
        bad_vals.to_csv(os.path.join(temp_dir, name), index=False)

        with pytest.raises(ValueError, match="files are the same length"):
            normalize.combine_run_metrics(temp_dir, "pulse_heights")

        # empty directory raises error
        empty_dir = os.path.join(temp_dir, "empty")
        os.makedirs(empty_dir)

        with pytest.raises(ValueError, match="No files"):
            normalize.combine_run_metrics(empty_dir, "pulse_heights")


@parametrize_with_cases("dir_names, mph_dfs, count_dfs", test_cases.TuningCurveFiles)
@pytest.mark.parametrize("count_range", [None, (0, 101)])
def test_combine_tuning_curve_metrics(dir_names, mph_dfs, count_dfs, count_range):
    with tempfile.TemporaryDirectory() as temp_dir:
        # variables to hold all unique values of each metric
        all_mph, all_counts, dir_paths, extreme_vals = [], [], [], []

        # create csv files with data to be combined
        for i in range(len(dir_names)):
            full_path = os.path.join(temp_dir, dir_names[i])
            os.makedirs(full_path)
            mph_dfs[i].to_csv(
                os.path.join(full_path, "fov-1-scan-1_pulse_heights.csv"), index=False
            )

            count_dfs[i].to_csv(
                os.path.join(full_path, "fov-1-scan-1_channel_counts.csv"), index=False
            )
            dir_paths.append(os.path.join(temp_dir, dir_names[i]))

            chan_min = count_dfs[i]["channel_count"].min()
            chan_max = count_dfs[i]["channel_count"].max()

            # if any values are extreme
            if count_range and (chan_min <= count_range[0] or chan_max >= count_range[1]):
                extreme_vals.append(dir_names[i])
            else:
                all_mph.extend(mph_dfs[i]["pulse_height"])
                all_counts.extend(count_dfs[i]["channel_count"])

        for fov in extreme_vals:
            dir_names.remove(fov)

        combined = normalize.combine_tuning_curve_metrics(dir_paths, count_range=count_range)

        # data may be in a different order due to matching dfs, but all values should be present
        assert set(all_mph) == set(combined["pulse_height"])
        assert set(all_counts) == set(combined["channel_count"])
        saved_dir_names = [name.split("/")[-1] for name in np.unique(combined["directory"])]
        assert set(saved_dir_names) == set(dir_names)

        # check that normalized value is 1 for maximum in each channel
        for mass in np.unique(combined["mass"]):
            subset = combined.loc[combined["mass"] == mass, :]
            max = np.max(subset[["channel_count"]].values)
            norm_vals = subset.loc[subset["channel_count"] == max, "norm_channel_count"].values
            assert np.all(norm_vals == 1)


def test_smooth_outliers():
    # Check for outliers which are separated by smoothing_range
    smooth_range = 2

    vals = np.arange(20, 40).astype("float")
    outlier1 = np.random.randint(3, 7)
    outlier2 = np.random.randint(9, 13)
    outlier3 = np.random.randint(15, 18)
    outliers = np.array([outlier1, outlier2, outlier3])
    smoothed_vals = normalize.smooth_outliers(
        vals=vals, outlier_idx=outliers, smooth_range=smooth_range
    )

    assert np.array_equal(vals, smoothed_vals)

    # check for outliers which are next to one another
    outliers = np.array([5, 6])
    smoothed_vals = normalize.smooth_outliers(
        vals=vals, outlier_idx=outliers, smooth_range=smooth_range
    )

    # 5th entry is two below, plus first two non-outliers above
    smooth_5 = np.mean(np.concatenate([vals[3:5], vals[7:9]]))

    # 6th entry is two below (one original and one smoothed from previous step), plus two above
    smooth_6 = np.mean(np.concatenate([vals[4:5], [smooth_5], vals[7:9]]))
    np.array_equal(smoothed_vals[outliers], [smooth_5, smooth_6])

    # check for outliers which are at the ends of the list
    outliers = np.array([0, 19])

    smoothed_vals = normalize.smooth_outliers(
        vals=vals, outlier_idx=outliers, smooth_range=smooth_range
    )
    # first entry is the mean of two above it
    outlier_0 = np.mean(vals[1:3])

    # second entry is mean of two below
    outlier_18 = np.mean(vals[17:19])

    assert np.allclose(smoothed_vals[outliers], np.array([outlier_0, outlier_18]))


def test_plot_voltage_vs_counts():
    with tempfile.TemporaryDirectory() as sweep_dir:
        folders = ["sweep_folder1", "sweep_folder2", "sweep_folder3", "sweep_folder4"]
        fov_paths = []
        voltages = [2000, 2020, 2040, 2060]

        for i, fov in enumerate(folders):
            fov_paths.append(os.path.join(sweep_dir, fov))
            os.makedirs(os.path.join(sweep_dir, fov))

            json_path = os.path.join(sweep_dir, fov, "fov-1-scan-1.json")
            json_data = {"hvDac": [{"name": "Detector", "currentSetPoint": voltages[i]}]}
            write_json_file(json_path, json_data)

        masses = [1, 2, 3]
        combined_data = pd.DataFrame(
            {
                "directory": ns.natsorted(fov_paths * len(masses)),
                "mass": masses * len(folders),
                "pulse_height": np.random.randint(1000, 3000, 12),
                "channel_count": np.random.randint(1000000, 3000000, 12),
            }
        )

        save_path = os.path.join(sweep_dir, "voltage_vs_counts.jpg")

        # test plot creation
        normalize.plot_voltage_vs_counts(fov_paths, combined_data, save_path)
        assert os.path.exists(save_path)


@patch("toffy.normalize.plt")
def test_show_multiple_plots(mock_plt):
    with tempfile.TemporaryDirectory() as temp_dir:
        plots = ["plot_1.jpg", "plot_2.jpg", "plot_3.jpg"]
        paths = [os.path.join(temp_dir, img) for img in plots]

        for img in plots:
            test_utils._make_small_file(temp_dir, img)

        normalize.show_multiple_plots(rows=2, cols=2, image_paths=paths)

        # check that the last image is loaded in
        mock_plt.imread.assert_called_with(os.path.join(temp_dir, "plot_3.jpg"))


def test_create_tuning_function(tmpdir, mocker):
    # create directory to hold the sweep
    sweep_dir = os.path.join(tmpdir, "sweep_1")
    os.makedirs(sweep_dir)

    # create individual runs each with a single FOV
    for voltage in [25, 50, 75]:
        run_dir = os.path.join(sweep_dir, "20220101_{}V".format(voltage))
        os.makedirs(run_dir)
        test_utils._make_blank_file(run_dir, "fov-1-scan-1.bin")
        json_path = os.path.join(run_dir, "fov-1-scan-1.json")
        json_data = {"hvDac": [{"name": "Detector", "currentSetPoint": voltage}]}
        write_json_file(json_path, json_data)

    # mock functions that interact with bin files directly
    mocker.patch("toffy.normalize.get_median_pulse_height", mocked_pulse_height)
    mocker.patch("toffy.normalize.extract_bin_files", mocked_extract_bin_file)
    mocker.patch("toffy.normalize.plt.show")

    # define paths for generated outputs
    save_path = os.path.join(tmpdir, "norm_func.json")
    plot_path = os.path.join(sweep_dir, "function_fit.jpg")
    all_plot_path = os.path.join(sweep_dir, "function_fit_all_data.jpg")

    # 3 runs should raise an error
    with pytest.raises(ValueError, match="Invalid amount of FOV folders"):
        normalize.create_tuning_function(sweep_path=sweep_dir, save_path=save_path)

    # create 4th run
    run_dir = os.path.join(sweep_dir, "20220101_{100V}")
    os.makedirs(run_dir)

    # missing bin file should raise an error
    with pytest.raises(ValueError, match="No bin file detected"):
        normalize.create_tuning_function(sweep_path=sweep_dir, save_path=save_path)

    test_utils._make_blank_file(run_dir, "fov-1-scan-1.bin")
    json_path = os.path.join(run_dir, "fov-1-scan-1.json")
    json_data = {"hvDac": [{"name": "Detector", "currentSetPoint": 100}]}
    write_json_file(json_path, json_data)

    # test success with count_range
    normalize.create_tuning_function(sweep_path=sweep_dir, save_path=save_path)
    assert os.path.exists(save_path)
    assert os.path.exists(plot_path)
    assert os.path.exists(all_plot_path)

    os.remove(save_path)
    os.remove(plot_path)
    os.remove(all_plot_path)

    # test success without count range
    normalize.create_tuning_function(sweep_path=sweep_dir, save_path=save_path, count_range=None)
    assert os.path.exists(save_path)
    assert os.path.exists(plot_path)
    assert not os.path.exists(all_plot_path)


def test_identify_outliers():
    # create dataset with specified outliers
    y_vals = np.arange(10, 30)
    x_vals = np.linspace(0, len(y_vals) - 1, len(y_vals))
    outlier_idx = [5, 10, 15]
    y_vals[outlier_idx] = [7, 32, 12]

    pred_outliers = normalize.identify_outliers(x_vals=x_vals, y_vals=y_vals, obj_func="poly_2")
    # check that outliers are correctly identified
    assert np.array_equal(outlier_idx, pred_outliers)


@parametrize("min_obs", [5, 12])
def test_fit_mass_mph_curve(tmpdir, min_obs):
    # create random data with single outlier
    mph_vals = np.random.randint(0, 3, 10) + np.arange(10)
    mph_vals[4] = 12

    mass_name = "88"
    obj_func = "poly_2"

    normalize.fit_mass_mph_curve(
        mph_vals=mph_vals,
        mass=mass_name,
        save_dir=tmpdir,
        obj_func=obj_func,
        min_obs=min_obs,
    )

    # make sure plot was created
    plot_path = os.path.join(tmpdir, mass_name + "_mph_fit.jpg")
    assert os.path.exists(plot_path)

    # make sure json with weights was created
    weights_path = os.path.join(tmpdir, mass_name + "_norm_func.json")

    mass_json = read_json_file(weights_path)

    # load weights into prediction function
    weights = mass_json["weights"]
    pred_func = normalize.create_prediction_function(name=obj_func, weights=weights)

    # generate predictions
    preds = pred_func(np.arange(10))

    if min_obs == 5:
        # check that prediction function generates unique output
        assert len(np.unique(preds)) == len(preds)
    else:
        # check that prediction function generates same output for all
        assert len(np.unique(preds)) == 1
        assert np.allclose(preds[0], np.median(mph_vals))


@parametrize("skip_norm_func", [False, True])
def test_create_fitted_mass_mph_vals(tmpdir, skip_norm_func):
    masses = ["88", "100", "120"]
    fovs = ["fov1", "fov2", "fov3", "fov4"]
    obj_func = "poly_2"

    # each mass has a unique multiplier for fitted function
    mass_mults = [1, 2, 3]

    # if skip_norm_func, pass on creating mass for 120
    end_idx = 2 if skip_norm_func else 3

    # create json for each channel
    for mass_idx in range(len(masses[:end_idx])):
        weights = [mass_mults[mass_idx], 0, 0]
        mass_json = {"name": obj_func, "weights": weights}
        mass_path = os.path.join(tmpdir, masses[mass_idx] + "_norm_func.json")

        write_json_file(json_path=mass_path, json_object=mass_json)

    # create combined mph_df
    pulse_height_list = np.random.rand(len(masses) * len(fovs))
    mass_list = np.tile(masses, len(fovs))
    fov_list = np.repeat(fovs, len(masses))

    pulse_height_df = pd.DataFrame(
        {"pulse_height": pulse_height_list, "mass": mass_list, "fov": fov_list}
    )

    modified_df = normalize.create_fitted_mass_mph_vals(
        pulse_height_df=pulse_height_df, obj_func_dir=tmpdir
    )

    # check that fitted values are correct multiplier of FOV
    for mass_idx in range(len(masses)):
        mass = masses[mass_idx]
        mult = mass_mults[mass_idx]

        fov_order = np.linspace(0, len(fovs) - 1, len(fovs))
        fitted_vals = modified_df.loc[modified_df["mass"] == mass, "pulse_height_fit"].values

        # special case for skip_norm_func and the final mass
        if skip_norm_func and mass_idx == len(masses) - 1:
            mult = 0

        assert np.array_equal(fov_order * mult, fitted_vals)


@parametrize("test_zeros", [False, True])
@parametrize("autogain", [False, True])
@parametrize_with_cases("metrics", cases=test_cases.CombineRunMetricFiles)
def test_create_fitted_pulse_heights_file(tmpdir, test_zeros, autogain, metrics):
    # create metric files
    pulse_dir = os.path.join(tmpdir, "pulse_heights")
    os.makedirs(pulse_dir)
    for metric in metrics["deficient"]:
        name, values_df = metric[0], pd.DataFrame(metric[1])

        # if test_zeros, set first mass pulse height to 0 for every FOV
        if test_zeros:
            values_df.iloc[0, 0] = 0

        values_df.to_csv(os.path.join(pulse_dir, name), index=False)

    for metric in metrics["proficient"]:
        name_prof, values_df_prof = metric[0], pd.DataFrame(metric[1])
        values_df_prof.to_csv(os.path.join(pulse_dir, name_prof), index=False)

    panel = test_cases.panel
    fovs = natsort.natsorted(test_cases.fovs)
    kwargs = {"min_obs": 4, "outlier_fraction": 2}

    if test_zeros:
        with pytest.warns(UserWarning, match="Skipping normalization"):
            df = normalize.create_fitted_pulse_heights_file(
                pulse_height_dir=pulse_dir,
                panel_info=panel,
                norm_dir=tmpdir,
                mass_obj_func="poly_3",
                autogain=autogain,
                **kwargs,
            )
    else:
        df = normalize.create_fitted_pulse_heights_file(
            pulse_height_dir=pulse_dir,
            panel_info=panel,
            norm_dir=tmpdir,
            mass_obj_func="poly_3",
            autogain=autogain,
            **kwargs,
        )

    # all four FOVs included
    assert len(np.unique(df["fov"].values)) == 5

    # FOVs are ordered in proper order
    ordered_fovs = df.loc[df["mass"] == 10, "fov"].values.astype("str")
    assert np.array_equal(ordered_fovs, fovs)

    # assert mass 5 is zero if test_zeros is True
    if test_zeros:
        assert np.all(df[df["mass"] == 5]["pulse_height"].values == 0)
        df = df[df["mass"] != 5].copy()

    # with autogain, max one FOV should contain the original value (depends on # of FOVs)
    if autogain:
        for mass in df["mass"].unique():
            mass_ph = df.loc[df["mass"] == mass, "pulse_height"].values
            mass_ph_med = np.median(mass_ph)
            assert np.sum(mass_ph[mass_ph == mass_ph_med]) <= 1
    # fitted values are distinct from original without autogain
    else:
        assert np.all(df["pulse_height"].values != df["pulse_height_fit"].values)


@parametrize("test_zeros", [False, True])
@parametrize("test_high_norm", [False, True])
@parametrize("test_low_norm", [False, True])
def test_normalize_fov(tmpdir, test_zeros, test_high_norm, test_low_norm):
    # create image data
    fovs, chans = test_utils.gen_fov_chan_names(num_fovs=1, num_chans=3)
    _, data_xr = test_utils.create_paired_xarray_fovs(
        tmpdir, fovs, chans, img_shape=(10, 10), dtype="float32"
    )

    # create inputs
    norm_vals = np.random.rand(len(chans))

    # need to ensure no extra low or high norm coefficients for testing
    norm_vals[norm_vals < 0.1] = 0.1
    norm_vals[norm_vals > 1.1] = 1.1

    # generate the actual normalization multipliers, needed for extreme normalization cap tests
    norm_mults = np.copy(norm_vals)

    # if test_zeros, set the first norm_val to 0 and the first channel to 0
    if test_zeros:
        norm_vals[0] = 0.0
        norm_mults[0] = 0.0
        data_xr[..., 0] = 0.0

    # if test_high_norm, set norm vals lower than 0.1
    if test_high_norm:
        norm_vals[1] = -0.01
        norm_mults[1] = 0.1

    # if test_low_norm, set norm vals higher than 1
    if test_low_norm:
        norm_vals[2] = 1.5
        norm_mults[2] = 1.1

    extreme_vals = (0.2, 1)
    norm_dir = os.path.join(tmpdir, "norm_dir")
    os.makedirs(norm_dir)

    if not test_high_norm or test_low_norm:
        normalize.normalize_fov(
            img_data=data_xr,
            norm_vals=norm_vals,
            norm_dir=norm_dir,
            fov=fovs[0],
            channels=chans,
            extreme_vals=extreme_vals,
        )
    else:
        if test_high_norm:
            with pytest.warns(UserWarning, match="are below 10% sensitivity"):
                normalize.normalize_fov(
                    img_data=data_xr,
                    norm_vals=norm_vals,
                    norm_dir=norm_dir,
                    fov=fovs[0],
                    channels=chans,
                    extreme_vals=extreme_vals,
                )
        if test_low_norm:
            with pytest.warns(UserWarning, match="will suffer a decrease"):
                normalize.normalize_fov(
                    img_data=data_xr,
                    norm_vals=norm_vals,
                    norm_dir=norm_dir,
                    fov=fovs[0],
                    channels=chans,
                    extreme_vals=extreme_vals,
                )

    norm_imgs = load_utils.load_imgs_from_tree(norm_dir, channels=chans)
    assert np.allclose(data_xr.values, norm_imgs.values * norm_mults)

    # check correct data type
    assert norm_imgs.dtype == data_xr.dtype

    # check that log file has correct values
    log_file = pd.read_csv(os.path.join(norm_dir, "fov0", "normalization_coefs.csv"))
    assert np.array_equal(log_file["channels"], chans)
    assert np.allclose(log_file["norm_vals"], norm_vals, rtol=1e-04)

    # check that warning is raised for extreme values
    with pytest.warns(UserWarning, match="inspection for accuracy is recommended"):
        norm_vals[0] = 0.19
        normalize.normalize_fov(
            img_data=data_xr,
            norm_vals=norm_vals,
            norm_dir=norm_dir,
            fov=fovs[0],
            channels=chans,
            extreme_vals=extreme_vals,
        )


@parametrize("autogain", [False, True])
@parametrize_with_cases("metrics", cases=test_cases.CombineRunMetricFiles)
def test_normalize_image_data(tmpdir, autogain, metrics):
    # create directory of pulse height csvs
    pulse_height_dir = os.path.join(tmpdir, "pulse_height_dir")
    os.makedirs(pulse_height_dir)

    for metric in metrics["deficient"]:
        name, values_df = metric[0], pd.DataFrame(metric[1])
        values_df.to_csv(os.path.join(pulse_height_dir, name), index=False)

    for metric in metrics["proficient"]:
        name_prof, values_df_prof = metric[0], pd.DataFrame(metric[1])
        values_df_prof.to_csv(os.path.join(pulse_height_dir, name_prof), index=False)

    # create directory with image data
    img_dir = os.path.join(tmpdir, "img_dir")
    os.makedirs(img_dir)

    fovs, chans = test_cases.fovs, test_cases.channels
    filelocs, data_xr = test_utils.create_paired_xarray_fovs(
        img_dir, fovs, chans, img_shape=(10, 10), dtype="float32"
    )

    # create mph norm func
    weights = np.random.rand(3)
    name = "poly_2"
    func_json = {"name": name, "weights": weights.tolist()}
    func_path = os.path.join(tmpdir, "norm_func.json")

    write_json_file(json_path=func_path, json_object=func_json)

    # get panel
    panel = test_cases.panel

    norm_dir = os.path.join(tmpdir, "norm_dir")
    os.makedirs(norm_dir)

    # normalize images
    normalize.normalize_image_data(
        img_dir=img_dir,
        norm_dir=norm_dir,
        pulse_height_dir=pulse_height_dir,
        panel_info=panel,
        norm_func_path=func_path,
        autogain=autogain,
    )

    assert np.array_equal(io_utils.list_folders(norm_dir, "fov").sort(), fovs.sort())

    # no normalization function
    with pytest.raises(ValueError, match="section 3 of the 1_set_up_toffy"):
        normalize.normalize_image_data(
            img_dir=img_dir,
            norm_dir=norm_dir,
            pulse_height_dir=pulse_height_dir,
            panel_info=panel,
            norm_func_path="bad_path",
            autogain=autogain,
        )

    # mismatch between FOVs
    shutil.rmtree(os.path.join(img_dir, fovs[0]))
    shutil.rmtree(norm_dir)
    os.makedirs(norm_dir)
    with pytest.raises(ValueError, match="image data fovs"):
        normalize.normalize_image_data(
            img_dir=img_dir,
            norm_dir=norm_dir,
            pulse_height_dir=pulse_height_dir,
            panel_info=panel,
            norm_func_path=func_path,
            autogain=autogain,
        )


def test_check_detector_voltage():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in list(range(1, 4)):
            fov_data = {"hvDac": [{"name": "Detector", "currentSetPoint": 1000}]}
            json_path = os.path.join(tmp_dir, "fov-" + str(i) + ".json")
            test_utils._make_blank_file(tmp_dir, "fov-" + str(i) + ".bin")
            write_json_file(json_path, fov_data)

        # constant voltage should run smoothly
        normalize.check_detector_voltage(tmp_dir)

        # skip empty json files
        test_utils._make_blank_file(tmp_dir, "fov-0.bin")
        test_utils._make_blank_file(tmp_dir, "fov-0.json")
        with pytest.warns(UserWarning, match="The following FOVs have empty json files"):
            normalize.check_detector_voltage(tmp_dir)

        for i in list(range(4, 11)):
            if i < 7:
                fov_data = {"hvDac": [{"name": "Detector", "currentSetPoint": 2000}]}
            else:
                fov_data = {"hvDac": [{"name": "Detector", "currentSetPoint": 3000}]}
            json_path = os.path.join(tmp_dir, "fov-" + str(i) + ".json")
            test_utils._make_blank_file(tmp_dir, "fov-" + str(i) + ".bin")
            write_json_file(json_path, fov_data)

        # change in voltage should raise an error
        with pytest.raises(
            ValueError,
            match=(
                "Between fov-3 and fov-4 the voltage changed from 1000 to 2000.\n"
                "Between fov-6 and fov-7 the voltage changed from 2000 to 3000."
            ),
        ):
            normalize.check_detector_voltage(tmp_dir)


@parametrize("detector_volt_type", ["hvAdc", "hvDac"])
def test_plot_detector_voltage(tmp_path_factory: TempPathFactory, detector_volt_type: str):
    run_folder: pathlib.Path = tmp_path_factory.mktemp("run_folder")
    mph_run_dir: pathlib.Path = tmp_path_factory.mktemp("mph_run_dir")
    json_files: List[str] = [f"fov-{i}-scan-1.json" for i in np.arange(1, 4)]

    for i, file in enumerate(json_files):
        sample_data = {
            "hvAdc": [{"name": "Detector", "value": 1 * (i + 1)}, {"name": "Not Detector"}],
            "hvDac": [
                {"name": "Detector", "currentSetPoint": 10 * (i + 1)},
                {"name": "Not Detector"},
            ],
        }

        write_json_file(run_folder / file, sample_data, encoding="utf-8")

    normalize.plot_detector_voltage(
        run_folder=run_folder, mph_run_dir=mph_run_dir, detector_volt_type=detector_volt_type
    )

    detector_gain_values_path: pathlib.Path = mph_run_dir / "detector_gain_values.csv"
    assert os.path.exists(detector_gain_values_path)

    detector_gain_values: pd.DataFrame = pd.read_csv(detector_gain_values_path)
    expected_voltage_vals: np.ndarray = (
        np.array([1, 2, 3]) if detector_volt_type == "hvAdc" else np.array([10, 20, 30])
    )
    assert np.all(detector_gain_values["FOVs"] == np.arange(1, 4))
    assert np.all(detector_gain_values["Detector Voltage"] == expected_voltage_vals)

    assert os.path.exists(mph_run_dir / "detector_gain_plot.png")
