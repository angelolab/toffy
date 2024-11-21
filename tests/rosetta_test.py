import copy
import os
import tempfile
import time
from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
import pytest
import skimage.io as io
from alpineer import image_utils, io_utils, load_utils, misc_utils, test_utils
from pytest_cases import parametrize_with_cases

from toffy import rosetta
from toffy.image_stitching import rescale_images
from toffy.rosetta import create_rosetta_matrices

from .utils import rosetta_test_cases as test_cases

parametrize = pytest.mark.parametrize


def test_compensate_matrix_simple():
    inputs = np.ones((2, 40, 40, 4))

    # each channel is an increasing multiple of original
    inputs[0, :, :, 1] *= 2
    inputs[0, :, :, 2] *= 3
    inputs[0, :, :, 3] *= 4

    # second fov is 10x greater than first
    inputs[1] = inputs[0] * 10

    # define coefficient matrix; each channel has a 2x higher multiplier than previous
    coeffs = np.array(
        [
            [0.01, 0, 0, 0.02],
            [0.02, 0, 0, 0.040],
            [0.04, 0, 0, 0.08],
            [0.08, 0, 0, 0.16],
        ]
    )

    # calculate amount that should be removed from first channel
    total_comp = (
        coeffs[0, 0] * inputs[0, 0, 0, 0]
        + coeffs[1, 0] * inputs[0, 0, 0, 1]
        + coeffs[2, 0] * inputs[0, 0, 0, 2]
        + coeffs[3, 0] * inputs[0, 0, 0, 3]
    )

    out_indices = np.arange(inputs.shape[-1])
    out = rosetta._compensate_matrix_simple(inputs, coeffs, out_indices)

    # non-affected channels are identical
    assert np.all(out[:, :, :, 1:-1] == inputs[:, :, :, 1:-1])

    # first channel is changed by baseline amount
    assert np.all(out[0, :, :, 0] == inputs[0, :, :, 0] - total_comp)

    # first channel in second fov is changed by baseline amount * 10 due to fov multiplier
    assert np.all(out[1, :, :, 0] == inputs[1, :, :, 0] - total_comp * 10)

    # last channel is changed by baseline amount * 2 due to multiplier in coefficient matrix
    assert np.all(out[0, :, :, -1] == inputs[0, :, :, -1] - total_comp * 2)

    # last channel in second fov is changed by baseline * 2 * 10 due to fov and coefficient
    assert np.all(out[1, :, :, -1] == inputs[1, :, :, -1] - total_comp * 10 * 2)

    # don't generate output for first channel
    out_indices = out_indices[1:]
    out = rosetta._compensate_matrix_simple(inputs, coeffs, out_indices)

    # non-affected channels are identical
    assert np.all(out[:, :, :, :-1] == inputs[:, :, :, 1:-1])

    # last channel is changed by baseline amount * 2 due to multiplier in coefficient matrix
    assert np.all(out[0, :, :, -1] == inputs[0, :, :, -1] - total_comp * 2)

    # last channel in second fov is changed by baseline * 2 * 10 due to fov and coefficient
    assert np.all(out[1, :, :, -1] == inputs[1, :, :, -1] - total_comp * 10 * 2)


def test_validate_inputs():
    with tempfile.TemporaryDirectory() as top_level_dir:
        data_dir = os.path.join(top_level_dir, "data_dir")
        os.makedirs(data_dir)

        # make fake data for testing
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=3)
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            data_dir, fovs, chans, img_shape=(10, 10), fills=True
        )

        # generate default, correct values for all parameters
        masses = [71, 76, 101]
        comp_mat_vals = np.random.rand(len(masses), len(masses)) / 100
        comp_mat = pd.DataFrame(comp_mat_vals, columns=masses, index=masses)

        acquired_masses = copy.copy(masses)
        acquired_targets = copy.copy(chans)
        input_masses = copy.copy(masses)
        output_masses = copy.copy(masses)
        all_masses = copy.copy(masses)
        save_format = "raw"
        raw_data_sub_folder = ""
        batch_size = 1
        gaus_rad = 1

        input_dict = {
            "raw_data_dir": data_dir,
            "comp_mat": comp_mat,
            "acquired_masses": acquired_masses,
            "acquired_targets": acquired_targets,
            "input_masses": input_masses,
            "output_masses": output_masses,
            "all_masses": all_masses,
            "fovs": fovs,
            "save_format": save_format,
            "raw_data_sub_folder": raw_data_sub_folder,
            "batch_size": batch_size,
            "gaus_rad": gaus_rad,
        }

        # check that masses are sorted
        input_dict_disorder = copy.copy(input_dict)
        input_dict_disorder["acquired_masses"] = masses[1:] + masses[:1]
        with pytest.raises(ValueError, match="Masses must be sorted"):
            rosetta.validate_inputs(**input_dict_disorder)

        # check that all masses are present
        input_dict_missing = copy.copy(input_dict)
        input_dict_missing["acquired_masses"] = masses[1:]
        with pytest.raises(ValueError, match="acquired masses and list compensation masses"):
            rosetta.validate_inputs(**input_dict_missing)

        # check that images and channels are the same
        input_dict_img_name = copy.copy(input_dict)
        input_dict_img_name["acquired_targets"] = chans + ["chan15"]
        with pytest.raises(ValueError, match="given in list listed channels"):
            rosetta.validate_inputs(**input_dict_img_name)

        # check that input masses are valid
        input_dict_input_mass = copy.copy(input_dict)
        input_dict_input_mass["input_masses"] = masses + [17]
        with pytest.raises(ValueError, match="list input masses"):
            rosetta.validate_inputs(**input_dict_input_mass)

        # check that output masses are valid
        input_dict_output_mass = copy.copy(input_dict)
        input_dict_output_mass["output_masses"] = masses + [17]
        with pytest.raises(ValueError, match="list output masses"):
            rosetta.validate_inputs(**input_dict_output_mass)

        # check that comp_mat has no NAs
        input_dict_na = copy.copy(input_dict)
        comp_mat_na = copy.copy(comp_mat)
        comp_mat_na.iloc[0, 2] = np.nan
        input_dict_na["comp_mat"] = comp_mat_na
        with pytest.raises(ValueError, match="no missing values"):
            rosetta.validate_inputs(**input_dict_na)

        # check that save_format is valid
        input_dict_save_format = copy.copy(input_dict)
        input_dict_save_format["save_format"] = "bad"
        with pytest.raises(ValueError, match="list save format"):
            rosetta.validate_inputs(**input_dict_save_format)

        # check that batch_size is valid
        input_dict_batch_size = copy.copy(input_dict)
        input_dict_batch_size["batch_size"] = 1.5
        with pytest.raises(ValueError, match="batch_size parameter"):
            rosetta.validate_inputs(**input_dict_batch_size)

        # check that gaus_rad is valid
        input_dict_gaus_rad = copy.copy(input_dict)
        input_dict_gaus_rad["gaus_rad"] = -1
        with pytest.raises(ValueError, match="gaus_rad parameter"):
            rosetta.validate_inputs(**input_dict_gaus_rad)


def test_clean_rosetta_test_dir():
    with tempfile.TemporaryDirectory() as rosetta_test_dir:
        # make a few dummy comp directories, each with a couple of FOV subdirectories
        mults = [0.5, 1, 2]
        fovs = ["fov0", "fov1"]
        comp_folder_names = []
        for m in mults:
            comp_folder_path = os.path.join(rosetta_test_dir, f"compensated_data_{m}")
            comp_folder_names.append(comp_folder_path)
            os.mkdir(comp_folder_path)

            for fov in fovs:
                os.mkdir(os.path.join(comp_folder_path, fov))

        # make a dummy stitched_images directory
        stitched_image_path = os.path.join(rosetta_test_dir, "stitched_images")
        os.mkdir(stitched_image_path)

        # make a few sample roseta matrix files
        source = "chan1"
        out = "chan2"
        rosetta_matrix_names = []
        for m in mults:
            mat_path = os.path.join(rosetta_test_dir, f"{source}_{out}_rosetta_matrix_mult_{m}.csv")
            rosetta_matrix_names.append(mat_path)
            pd.DataFrame().to_csv(mat_path)

        # make example ._ files to simulate external drives
        Path(os.path.join(rosetta_test_dir, "compensated_data_%s" % mults[0], "._random")).touch()
        Path(os.path.join(rosetta_test_dir, "stitched_images", "._random")).touch()

        # run the cleaning process
        rosetta.clean_rosetta_test_dir(rosetta_test_dir)

        # assert all of the comp directories are deleted
        for cfn in comp_folder_names:
            assert not os.path.exists(cfn)

        # assert the stitched image directories is deleted
        assert not os.path.exists(stitched_image_path)

        # assert the rosetta matrices still exist
        for rmn in rosetta_matrix_names:
            assert os.path.exists(rmn)

        # ensure no ._ files remain
        rosetta_test_files = Path(rosetta_test_dir)
        rosetta_test_files = [str(f) for f in list(Path(rosetta_test_files).rglob("*"))]
        assert not any(["._" in f for f in rosetta_test_files])


def test_combine_compensation_files():
    with tempfile.TemporaryDirectory() as rosetta_test_dir:
        # make a few dummy matrix files for different multipliers and channels
        mults = [0.5, 1, 2]
        channels = ["chan1", "chan2", "chan3"]
        channel_pairs = [("chan1", "chan2"), ("chan2", "chan3"), ("chan3", "chan1")]

        for m in mults:
            for cp in channel_pairs:
                df = pd.DataFrame(
                    np.zeros((3, 4)),
                    index=channels,
                    columns=[None] + channels,
                )
                df.iloc[:, 0] = np.arange(3)
                df.loc[cp[0], cp[1]] = m
                df.to_csv(
                    os.path.join(rosetta_test_dir, f"{cp[0]}_{cp[1]}_compensation_matrix_{m}.csv"),
                    index=False,
                )

        # combine the chan1-chan2 x0.5, chan2-chan3 x1, and chan3-chan1 x2 matrix
        compensation_matrices = [
            "chan1_chan2_compensation_matrix_0.5.csv",
            "chan2_chan3_compensation_matrix_1.csv",
            "chan3_chan1_compensation_matrix_2.csv",
        ]
        rosetta.combine_compensation_files(
            rosetta_test_dir, compensation_matrices, "final_rosetta_matrix.csv"
        )

        # assert final_rosetta_matrix.csv created
        final_rosetta_matrix_path = os.path.join(rosetta_test_dir, "final_rosetta_matrix.csv")
        assert os.path.exists(final_rosetta_matrix_path)

        # assert the compensation coefficients got combined correctly
        final_rosetta_matrix = pd.read_csv(final_rosetta_matrix_path)
        actual_final_values = np.array([[0, 0.5, 0], [0, 0, 1], [2, 0, 0]])
        assert np.all(final_rosetta_matrix.values[:, 1:] == actual_final_values)

        # assert the channel column didn't get modified
        actual_column_values = np.arange(3)
        assert np.all(final_rosetta_matrix.iloc[:, 0].values == actual_column_values)


def test_flat_field_correction():
    input_img = np.random.rand(10, 10)
    corrected_img = rosetta.flat_field_correction(img=input_img)

    assert corrected_img.shape == input_img.shape
    assert not np.array_equal(corrected_img, input_img)

    # test empty image
    with pytest.warns(UserWarning, match="Image for flatfield correction is empty"):
        input_img = np.zeros((10, 10))
        corrected_img = rosetta.flat_field_correction(img=input_img)

        assert not np.any(corrected_img)


def test_get_masses_from_channel_names():
    targets = ["chan1", "chan2", "chan3"]
    masses = [1, 2, 3]
    test_df = pd.DataFrame({"Target": targets, "Mass": masses})

    all_masses = rosetta.get_masses_from_channel_names(targets, test_df)
    assert np.array_equal(masses, all_masses)

    subset_masses = rosetta.get_masses_from_channel_names(targets[:2], test_df)
    assert np.array_equal(masses[:2], subset_masses)

    with pytest.raises(ValueError, match="channel names"):
        rosetta.get_masses_from_channel_names(["chan4"], test_df)


@parametrize("output_masses", [None, [25, 50, 101], [25, 50]])
@parametrize("input_masses", [None, [25, 50, 101], [25, 50]])
@parametrize("gaus_rad", [0, 1, 2])
@parametrize("save_format", ["raw", "rescaled", "both"])
@parametrize("ffc_masses", [None, [50]])
@parametrize_with_cases("panel_info", cases=test_cases.CompensateImageDataPanel)
@parametrize_with_cases("comp_mat", cases=test_cases.CompensateImageDataMat)
def test_compensate_image_data(
    output_masses, input_masses, gaus_rad, save_format, panel_info, comp_mat, ffc_masses
):
    with tempfile.TemporaryDirectory() as top_level_dir:
        data_dir = os.path.join(top_level_dir, "data_dir")
        output_dir = os.path.join(top_level_dir, "output_dir")

        os.makedirs(data_dir)
        os.makedirs(output_dir)

        # make fake data for testing
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=3)
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            data_dir, fovs, chans, img_shape=(10, 10), fills=True, dtype=np.uint32
        )
        os.makedirs(os.path.join(data_dir, "stitched_images"))

        # create compensation matrix
        comp_mat_path = os.path.join(data_dir, "comp_mat.csv")
        comp_mat.to_csv(comp_mat_path)

        # call function
        rosetta.compensate_image_data(
            data_dir,
            output_dir,
            comp_mat_path,
            panel_info,
            input_masses=input_masses,
            output_masses=output_masses,
            save_format=save_format,
            gaus_rad=gaus_rad,
            ffc_masses=ffc_masses,
            correct_streaks=True,
            streak_chan="chan1",
        )

        # all folders created
        output_folders = io_utils.list_folders(output_dir)
        assert set(fovs) == set(output_folders)

        # determine output directory structure
        format_folders = ["raw", "rescaled"]
        if save_format in format_folders:
            format_folders = [save_format]

        for folder in format_folders:
            # check that all files were created
            output_files = io_utils.list_files(os.path.join(output_dir, fovs[0], folder), ".tiff")
            output_files = [chan.split(".tiff")[0] for chan in output_files]

            if output_masses is None or len(output_masses) == 3:
                assert set(output_files) == set(panel_info["Target"].values)
            else:
                assert set(output_files) == set(panel_info["Target"].values[:-1])

            output_data = load_utils.load_imgs_from_tree(data_dir=output_dir, img_sub_folder=folder)

            assert np.issubdtype(output_data.dtype, np.floating)

            # all channels are smaller than original
            for i in range(output_data.shape[0]):
                for j in range(output_data.shape[-1]):
                    assert np.sum(output_data.values[i, :, :, j]) <= np.sum(
                        data_xr.values[i, :, :, j]
                    )


def test_copy_round_one_compensated_images():
    with tempfile.TemporaryDirectory() as top_level_dir:
        round_one_folder = os.path.join(top_level_dir, "round_one")
        round_two_folder = os.path.join(top_level_dir, "round_two")
        runs = ["run1", "run2"]
        fovs = ["fov-1-scan-1", "fov-2-scan-2"]
        channel_list = ["chan1", "chan2", "chan3"]

        for run in runs:
            os.makedirs(os.path.join(round_one_folder, run))
            os.makedirs(os.path.join(round_two_folder, run))

            for fov in fovs:
                # create sample FOV folders
                os.makedirs(os.path.join(round_one_folder, run, fov, "rescaled"))
                os.makedirs(os.path.join(round_two_folder, run, fov, "rescaled"))

                # create sample channel files, include one that has already been written to round 2
                channel_list = ["chan1", "chan2", "chan3"]
                for chan in channel_list:
                    Path(
                        os.path.join(round_one_folder, run, fov, "rescaled", chan + ".tiff")
                    ).touch()

                Path(
                    os.path.join(round_two_folder, run, fov, "rescaled", channel_list[-1] + ".tiff")
                ).touch()

        # copy the images and assert all channels exist in round two folder
        rosetta.copy_round_one_compensated_images(
            runs, round_one_folder, round_two_folder, channel_list[:2]
        )

        for run in runs:
            for fov in fovs:
                chans_in_folder = io_utils.list_files(
                    os.path.join(round_two_folder, run, fov, "rescaled"), substrs=".tiff"
                )
                chans_in_folder = [c.replace(".tiff", "") for c in chans_in_folder]

                misc_utils.verify_same_elements(
                    all_channels=channel_list, channels_in_folder=chans_in_folder
                )


@parametrize("dir_num", [2, 3])
@parametrize("channel_subset", [False, True])
@pytest.mark.parametrize("img_size_scale", [None, 0.5])
def test_create_tiled_comparison(dir_num, channel_subset, img_size_scale):
    with tempfile.TemporaryDirectory() as top_level_dir:
        img_scale = img_size_scale if img_size_scale else 1
        num_chans = 3
        num_fovs = 4

        output_dir = os.path.join(top_level_dir, "output_dir")
        os.makedirs(output_dir)
        dir_names = ["input_dir_{}".format(i) for i in range(dir_num)]

        # create matching input directories
        for input_dir in dir_names:
            full_path = os.path.join(top_level_dir, input_dir)
            os.makedirs(full_path)

            fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs, num_chans=num_chans)
            filelocs, data_xr = test_utils.create_paired_xarray_fovs(
                full_path,
                fovs,
                chans,
                img_shape=(10, 10),
                fills=True,
                sub_dir="rescaled",
            )

        # pass full paths to function
        # NOTE: channel_subset tests for some and all channels provided
        paths = [os.path.join(top_level_dir, img_dir) for img_dir in dir_names]
        chan_list = chans[:-1] if channel_subset else chans[:]
        rosetta.create_tiled_comparison(
            paths,
            output_dir,
            max_img_size=10,
            channels=chan_list if channel_subset else None,
            img_size_scale=img_scale,
        )

        # check that each tiled image was created
        for i in range(len(chan_list)):
            chan_name = "chan{}_comparison.tiff".format(i)
            chan_img = io.imread(os.path.join(output_dir, chan_name))
            row_len = num_fovs * 10 * img_scale
            col_len = dir_num * 10 * img_scale
            assert chan_img.shape == (col_len, row_len)

        # check that directories with different images are okay if overlapping channels specified
        for i in range(num_fovs):
            os.remove(
                os.path.join(
                    top_level_dir,
                    dir_names[1],
                    "fov{}".format(i),
                    "rescaled/chan0.tiff",
                )
            )

        # no error raised if subset directory is specified
        rosetta.create_tiled_comparison(
            paths,
            output_dir,
            channels=["chan1", "chan2"],
            max_img_size=10,
            img_size_scale=img_scale,
        )

        # but one is raised if no subset directory is specified
        with pytest.raises(ValueError, match="1 of 1"):
            rosetta.create_tiled_comparison(paths, output_dir, max_img_size=10)


@parametrize("percent_norm", [98, None])
@parametrize("img_scale", [1, 0.25])
def test_add_source_channel_to_tiled_image(percent_norm, img_scale):
    with tempfile.TemporaryDirectory() as top_level_dir:
        num_fovs = 5
        num_chans = 4
        im_size = 12

        # create directory containing raw images
        raw_dir = os.path.join(top_level_dir, "raw_dir")
        os.makedirs(raw_dir)

        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs, num_chans=num_chans)
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            raw_dir, fovs, chans, img_shape=(im_size, im_size), fills=True
        )

        # create directory containing stitched images
        tiled_shape = (im_size * 3, im_size * num_fovs)
        tiled_dir = os.path.join(top_level_dir, "tiled_dir")
        os.makedirs(tiled_dir)
        for i in range(2):
            vals = np.random.rand(im_size * 3 * im_size * num_fovs).reshape(tiled_shape)
            if img_scale != 1:
                vals = rescale_images(vals, scale=img_scale)
            fname = os.path.join(tiled_dir, f"tiled_image_{i}.tiff")
            image_utils.save_image(fname, vals)

        output_dir = os.path.join(top_level_dir, "output_dir")
        os.makedirs(output_dir)
        rosetta.add_source_channel_to_tiled_image(
            raw_img_dir=raw_dir,
            tiled_img_dir=tiled_dir,
            output_dir=output_dir,
            source_channel="chan1",
            max_img_size=im_size,
            percent_norm=percent_norm,
            img_size_scale=img_scale,
        )

        # each image should now have an extra row added on top
        tiled_images = io_utils.list_files(output_dir)
        for im_name in tiled_images:
            image = io.imread(os.path.join(output_dir, im_name))
            assert image.shape == (
                img_scale * (tiled_shape[0] + im_size),
                img_scale * tiled_shape[1],
            )


@parametrize("fovs", [None, ["fov1"]])
@parametrize("replace", [True, False])
def test_replace_with_intensity_image(replace, fovs):
    with tempfile.TemporaryDirectory() as top_level_dir:
        # create directory containing raw images
        run_dir = os.path.join(top_level_dir, "run_dir")
        os.makedirs(run_dir)

        fov_names, chans = test_utils.gen_fov_chan_names(num_fovs=2, num_chans=2)
        chans = [chan + "_intensity" for chan in chans]
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            run_dir,
            fov_names,
            chans,
            img_shape=(10, 10),
            fills=True,
            sub_dir="intensities",
        )

        rosetta.replace_with_intensity_image(
            run_dir=run_dir, channel="chan1", replace=replace, fovs=fovs
        )

        # loop through all fovs to check that correct image was written
        for current_fov in range(2):
            if fovs is not None and current_fov == 0:
                # this fov was skipped, no images should be present here
                files = io_utils.list_files(os.path.join(run_dir, "fov0"))
                assert len(files) == 0
            else:
                # ensure correct extension is present
                if replace:
                    suffix = ".tiff"
                else:
                    suffix = "_intensity.tiff"
                file = os.path.join(run_dir, "fov{}".format(current_fov), "chan1" + suffix)
                assert os.path.exists(file)


def test_remove_sub_dirs():
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs = ["fov1", "fov2", "fov3"]
        sub_dirs = ["sub1", "sub2", "sub3"]

        # make directory structure
        for fov in fovs:
            os.makedirs(os.path.join(temp_dir, fov))
            for sub_dir in sub_dirs:
                os.makedirs(os.path.join(temp_dir, fov, sub_dir))

        rosetta.remove_sub_dirs(run_dir=temp_dir, sub_dirs=sub_dirs[1:], fovs=fovs[:-1])

        # check that last fov has all sub_dirs, all other fovs have appropriate sub_dirs removed
        for fov in fovs:
            if fov == fovs[-1]:
                expected_dirs = sub_dirs
            else:
                expected_dirs = sub_dirs[:1]

            for sub_dir in sub_dirs:
                if sub_dir in expected_dirs:
                    assert os.path.exists(os.path.join(temp_dir, fov, sub_dir))
                else:
                    assert not os.path.exists(os.path.join(temp_dir, fov, sub_dir))


def test_create_rosetta_matrices():
    with tempfile.TemporaryDirectory() as temp_dir:
        # create baseline rosetta matrix
        test_channels = [23, 71, 89, 113, 141, 142, 143]
        base_matrix = np.random.randint(1, 50, size=[len(test_channels), len(test_channels)])
        base_rosetta = pd.DataFrame(base_matrix, index=test_channels, columns=test_channels)
        base_rosetta_path = os.path.join(temp_dir, "rosetta_matrix.csv")
        base_rosetta.to_csv(base_rosetta_path)

        # validate output when all channels are included
        # NOTE: current_channel_name and output_channel_names are dummies, only for naming files
        multipliers = [0.5, 2, 4]
        create_rosetta_matrices(
            base_rosetta_path,
            temp_dir,
            multipliers,
            current_channel_name="chan1",
            output_channel_names=["chan2", "chan3"],
        )

        for multiplier in multipliers:
            rosetta_path = os.path.join(
                temp_dir,
                "chan1_chan2_chan3_rosetta_matrix_mult_%s.csv" % (str(multiplier)),
            )
            # grabs output of create_rosetta_matrices
            test_matrix = pd.read_csv(rosetta_path, index_col=0)
            rescaled = test_matrix / multiplier

            # confirm all channels scaled by multiplier
            assert np.array_equal(base_rosetta, rescaled)

        # validate output for specific channels
        mod_channels = [113, 142]
        create_rosetta_matrices(
            base_rosetta_path,
            temp_dir,
            multipliers,
            current_channel_name="chan1",
            output_channel_names=["chan2", "chan3"],
            masses=mod_channels,
        )

        # create mask specifying which channels will change
        change_idx = np.isin(test_channels, mod_channels)

        for mult in multipliers:
            mult_vec = np.ones(len(test_channels))
            mult_vec[change_idx] = mult

            # grabs output of create_rosetta_matrices
            rosetta_path = os.path.join(
                temp_dir, "chan1_chan2_chan3_rosetta_matrix_mult_%s.csv" % (str(mult))
            )
            test_matrix = pd.read_csv(rosetta_path, index_col=0)

            rescaled = test_matrix.divide(mult_vec, axis="index")
            assert np.array_equal(base_rosetta, rescaled)

        # check that error is raised when non-numeric rosetta_matrix is passed
        bad_matrix = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": ["a", "b", "c"]})
        bad_matrix_path = os.path.join(temp_dir, "bad_rosetta_matrix.csv")
        bad_matrix.to_csv(bad_matrix_path)

        with pytest.raises(ValueError, match="include only numeric"):
            create_rosetta_matrices(
                bad_matrix_path,
                temp_dir,
                multipliers,
                current_channel_name="chan1",
                output_channel_names=["chan2", "chan3"],
            )


def mock_img_size(run_dir, fov_list=None):
    run = os.path.basename(run_dir)
    sizes = {"run_1": 16, "run_2": 32, "run_3": 16}
    return sizes[run]


# TODO: anything with [f for f in os.listdir(...) ...] needs to be changed
# after list_folders with substrs specified is fixed
def test_copy_image_files(mocker):
    mocker.patch("toffy.image_stitching.get_max_img_size", mock_img_size)

    run_names = ["run_1", "run_2", "run_3"]
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(0, 3):
            run_folder = os.path.join(temp_dir, run_names[i])
            os.makedirs(run_folder)
            for j in range(1, 6):
                if (j < 5) or (j == 5 and i < 2):
                    os.makedirs(os.path.join(run_folder, f"fov-{j}"))
                    test_utils._make_blank_file(
                        os.path.join(run_folder, f"fov-{j}"), "test_image.tif"
                    )
            os.makedirs(os.path.join(run_folder, "stitched_images"))

        with tempfile.TemporaryDirectory() as temp_dir2:
            # bad run name should raise an error
            with pytest.raises(ValueError, match="not a valid run name"):
                rosetta.copy_image_files("cohort_name", ["bad_name"], temp_dir2, temp_dir)

            # bad paths should raise an error
            with pytest.raises(FileNotFoundError, match="could not be found"):
                rosetta.copy_image_files("cohort_name", run_names, "bad_path", temp_dir)
                rosetta.copy_image_files("cohort_name", run_names, temp_dir2, "bad_path")

            # not enough fov files for provided arg
            with pytest.raises(ValueError, match="contain the minimum amount of FOVs"):
                rosetta.copy_image_files(
                    "cohort_name", run_names, temp_dir2, temp_dir, fovs_per_run=10
                )

            # test successful folder copy
            rosetta.copy_image_files("cohort_name", run_names, temp_dir2, temp_dir, fovs_per_run=4)

            # check that correct total and per run fovs are copied
            extracted_fov_dir = os.path.join(temp_dir2, "cohort_name", "extracted_images")
            assert len(io_utils.list_folders(extracted_fov_dir)) == 12
            for i in range(1, 4):
                assert len([f for f in os.listdir(extracted_fov_dir) if f"run_{i}" in f]) == 4
            assert len(list(io_utils.list_folders(extracted_fov_dir, "stitched_images"))) == 0

            # check that files in fov folders are copied
            for folder in io_utils.list_folders(extracted_fov_dir):
                assert os.path.exists(os.path.join(extracted_fov_dir, folder, "test_image.tif"))

            # test successful folder copy with some runs skipped
            rmtree(os.path.join(temp_dir2, "cohort_name"))
            with pytest.warns(UserWarning, match="The following runs will be skipped"):
                rosetta.copy_image_files(
                    "cohort_name", run_names, temp_dir2, temp_dir, fovs_per_run=5
                )

            # check that correct total and per run fovs are copied, assert run 3 didn't get copied
            assert len(io_utils.list_folders(extracted_fov_dir)) == 10
            for i in range(1, 3):
                assert len([f for f in os.listdir(extracted_fov_dir) if f"run_{i}" in f]) == 5
            assert len([f for f in os.listdir(extracted_fov_dir) if f"run_3" in f]) == 0

            # check that files in fov folders are copied
            for folder in io_utils.list_folders(extracted_fov_dir):
                assert os.path.exists(os.path.join(extracted_fov_dir, folder, "test_image.tif"))


def test_rescale_raw_imgs():
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs = ["fov-1-scan1", "fov-2-scan-1"]
        channels = ["Au", "CD3"]
        test_utils._write_tifs(
            base_dir=temp_dir,
            fov_names=fovs,
            img_names=channels,
            shape=(32, 32),
            sub_dir="",
            fills=False,
            dtype="uint32",
        )
        img_data = load_utils.load_imgs_from_tree(temp_dir)

        # bad extracted img path should raise an error
        with pytest.raises(FileNotFoundError):
            rosetta.rescale_raw_imgs("bad_path")

        # test successful image saving
        rosetta.rescale_raw_imgs(temp_dir)
        for i in range(0, 2):
            img_path = os.path.join(temp_dir, fovs[i], "rescaled", channels[i] + ".tiff")
            assert os.path.exists(img_path)

        # test successful rescale of data
        rescaled_img_data = load_utils.load_imgs_from_tree(temp_dir, "rescaled")
        create_time = Path(os.path.join(temp_dir, fovs[0], "rescaled")).stat().st_ctime
        assert rescaled_img_data.all() == (img_data / 200).all()

        assert rescaled_img_data.dtype == "float32"

        # re-run function and check the files are not re-written
        time.sleep(3)
        rosetta.rescale_raw_imgs(temp_dir)
        modify_time = Path(os.path.join(temp_dir, fovs[0], "rescaled")).stat().st_mtime
        assert np.isclose(modify_time, create_time)


def create_rosetta_comp_structure(
    raw_data_dir,
    comp_data_dir,
    comp_mat_path,
    panel_info,
    input_masses=None,
    output_masses=None,
    save_format="rescaled",
    raw_data_sub_folder="",
    batch_size=1,
    gaus_rad=1,
    norm_const=200,
    ffc_masses=[39],
    correct_streaks=False,
    streak_chan="Noodle",
):
    fovs = ["fov-1-scan-1", "fov-2-scan-1"]
    channels = ["chan_39", "Noodle", "Au"]
    if output_masses:
        channels = ["Au"]

    for i in range(0, len(fovs)):
        os.makedirs(os.path.join(comp_data_dir, fovs[i], save_format))
        for j in range(0, len(channels)):
            test_utils._make_blank_file(
                os.path.join(comp_data_dir, fovs[i], save_format), channels[j] + ".tiff"
            )


def test_generate_rosetta_test_imgs(mocker):
    mocker.patch("toffy.rosetta.compensate_image_data", create_rosetta_comp_structure)

    with tempfile.TemporaryDirectory() as temp_img_dir:
        fovs = ["fov-1-scan-1", "fov-2-scan-1"]
        channels = ["chan_39", "Noodle", "Au"]
        test_utils._write_tifs(
            base_dir=temp_img_dir,
            fov_names=fovs,
            img_names=channels,
            shape=(32, 32),
            sub_dir="",
            fills=False,
            dtype="uint32",
        )

        rosetta_mat_path = os.path.join(
            Path(__file__).parent.parent, "files", "commercial_rosetta_matrix_v1.csv"
        )
        panel = pd.DataFrame(
            {
                "Mass": [39, 117, 197],
                "Target": channels,
                "Start": [38.7, 116.7, 196.7],
                "Stop": [39, 117, 197],
            }
        )
        mults = [0.5, 1]

        with tempfile.TemporaryDirectory() as temp_dir:
            # bad paths should raise error
            with pytest.raises(FileNotFoundError):
                rosetta.generate_rosetta_test_imgs("bad_path", temp_img_dir, mults, temp_dir, panel)
                rosetta.generate_rosetta_test_imgs(
                    rosetta_mat_path, "bad_path", mults, temp_dir, panel
                )
                rosetta.generate_rosetta_test_imgs(
                    rosetta_mat_path, temp_img_dir, mults, "bad_path", panel
                )

            # test success for all channels
            rosetta.generate_rosetta_test_imgs(
                rosetta_mat_path, temp_img_dir, mults, temp_dir, panel
            )

            for i, j, k in zip(mults, range(0, len(fovs)), range(0, len(channels))):
                assert os.path.exists(
                    os.path.join(
                        temp_dir,
                        f"compensated_data_{i}",
                        fovs[j],
                        "rescaled",
                        channels[k] + ".tiff",
                    )
                )
                assert os.path.exists(
                    os.path.join(
                        temp_dir,
                        f"Noodle_all_commercial_rosetta_matrix_v1_mult_{i}.csv",
                    )
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            # test success for subset of  channels
            one_channel = ["Au"]

            rosetta.generate_rosetta_test_imgs(
                rosetta_mat_path,
                temp_img_dir,
                mults,
                temp_dir,
                panel,
                output_channel_names=one_channel,
            )

            for i, j in zip(mults, range(0, len(fovs))):
                assert os.path.exists(
                    os.path.join(
                        temp_dir,
                        f"compensated_data_{i}",
                        fovs[j],
                        "rescaled",
                        "Au.tiff",
                    )
                )
                assert os.path.exists(
                    os.path.join(temp_dir, f"Noodle_Au_commercial_rosetta_matrix_v1_mult_{i}.csv")
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            # test if ffc_masses is None
            rosetta.generate_rosetta_test_imgs(
                rosetta_mat_path, temp_img_dir, mults, temp_dir, panel, ffc_masses=None
            )
