from typing import Tuple
import pytest
from pathlib import Path
from toffy import streak_detection as sd
import numpy as np
import pandas as pd
from skimage import io
from collections import namedtuple
import xarray as xr
from ark.utils import test_utils
from functools import partial


@pytest.fixture(scope="function")
def streak_dataset():
    def streak_data_generator(corrected_dir, fov: str, chan: str, shape: Tuple[int, int]):
        _test_data = sd.StreakData(
            shape=shape,
            fov=fov,
            corrected_dir=corrected_dir,
            streak_channel=chan,
            streak_mask=np.zeros(shape=shape),
            streak_df=pd.util.testing.makeDataFrame(),
            filtered_streak_mask=np.zeros(shape=shape),
            filtered_streak_df=pd.util.testing.makeDataFrame(),
            boxed_streaks=np.zeros(shape=shape),
            corrected_streak_mask=np.zeros(shape=shape),
        )
        return _test_data

    return streak_data_generator


@pytest.fixture(scope="function")
def df_dataset():
    def test_df_generator(shape: Tuple[int, int], df_row_num: int):
        col_size = shape[0]
        row_size = shape[1]

        # Initialize a new generator - set seed for reproducibility
        rng = np.random.default_rng(12345)

        # Create test_df
        min_row = rng.integers(low=0, high=row_size - 1, size=(df_row_num))
        max_row = min_row + 1
        min_col = rng.integers(low=0, high=col_size - 1, size=(df_row_num))
        max_col = rng.integers(low=min_col, high=row_size - 1, size=(df_row_num))
        test_df = pd.DataFrame(
            np.stack([min_row, max_row, min_col, max_col], axis=-1),
            columns=["min_row", "max_row", "min_col", "max_col"],
        )
        return test_df

    return test_df_generator


def test_get_save_dir(tmp_path):
    data_dir: Path = Path(tmp_path)
    name: str = "streak_data_test"
    ext_csv: str = "csv"
    ext_tiff: str = "tiff"

    csv_save_dir: Path = sd._get_save_dir(data_dir=data_dir, name=name, ext=ext_csv)
    assert csv_save_dir == Path(tmp_path / "streak_data_test.csv")

    tiff_save_dir: Path = sd._get_save_dir(data_dir=data_dir, name=name, ext=ext_tiff)
    assert tiff_save_dir == Path(tmp_path / "streak_data_test.tiff")


def test_save(tmp_path, streak_dataset):
    # Create minimum data needed for testing the save functionality with dataframes and numpy array
    # streak_data needs: corrected_dir, streak_channel, and data (streak_mask, streak_df,
    # filtered_streak_mask, filtered_streak_df, boxed_streaks, corrected_streak_mask)

    streak_data_test = streak_dataset(
        corrected_dir=tmp_path, fov="fov0", chan="chan0", shape=(1000, 1000)
    )

    Read_Fn = namedtuple("Read_Operation", "read_fn, ext")
    fields = {
        "streak_mask": Read_Fn(io.imread, ".tiff"),
        "streak_df": Read_Fn(pd.read_csv, ".csv"),
        "filtered_streak_mask": Read_Fn(io.imread, ".tiff"),
        "filtered_streak_df": Read_Fn(pd.read_csv, ".csv"),
        "boxed_streaks": Read_Fn(io.imread, ".tiff"),
        "corrected_streak_mask": Read_Fn(io.imread, ".tiff"),
    }

    #  Test `streak_detection::_save`
    for field, rf in fields.items():
        sd._save(streak_data=streak_data_test, name=field)

        if rf.ext == ".csv":
            data = rf.read_fn(Path(tmp_path, f"streak_data_chan0", field + rf.ext), index_col=0)
        else:
            data = rf.read_fn(Path(tmp_path, f"streak_data_chan0", field + rf.ext))

        if type(data) is np.ndarray:
            assert np.array_equal(data, getattr(streak_data_test, field))
        if type(data) is pd.DataFrame:
            pd.testing.assert_frame_equal(data, getattr(streak_data_test, field))


def test_save_streak_masks(tmp_path, streak_dataset):
    # * Minimum Data Needed:
    #   1. streak_data with the fields: corrected_dir, streak_channel, and data (streak_mask,
    #   streak_df, filtered_streak_mask, filtered_streak_df, boxed_streaks, corrected_streak_mask)

    streak_data_test = streak_dataset(
        corrected_dir=tmp_path, fov="fov0", chan="chan0", shape=(1000, 1000)
    )

    sd._save_streak_masks(streak_data=streak_data_test)

    Read_Fn = namedtuple("Read_Operation", "read_fn, ext")
    fields = {
        "streak_mask": Read_Fn(io.imread, ".tiff"),
        "streak_df": Read_Fn(pd.read_csv, ".csv"),
        "filtered_streak_mask": Read_Fn(io.imread, ".tiff"),
        "filtered_streak_df": Read_Fn(pd.read_csv, ".csv"),
        "boxed_streaks": Read_Fn(io.imread, ".tiff"),
        "corrected_streak_mask": Read_Fn(io.imread, ".tiff"),
    }

    #  Test `streak_detection::_save`
    for field, rf in fields.items():
        if rf.ext == ".csv":
            data = rf.read_fn(Path(tmp_path, f"streak_data_chan0", field + rf.ext), index_col=0)
        else:
            data = rf.read_fn(Path(tmp_path, f"streak_data_chan0", field + rf.ext))

        if type(data) is np.ndarray:
            assert np.array_equal(data, getattr(streak_data_test, field))
        if type(data) is pd.DataFrame:
            pd.testing.assert_frame_equal(data, getattr(streak_data_test, field))


def test_make_binary_mask():
    # * Minimum Data Needed
    #   1. Numpy array
    # Only need to test to make sure the dimensions are the same, and the output is a binary array
    rng = np.random.default_rng(12345)
    row_size = 1000
    col_size = 1000
    input_image_test = rng.integers(low=0, high=16, size=(row_size, col_size))

    binary_mask = sd._make_binary_mask(input_image=input_image_test)
    assert np.all(np.isin(binary_mask, [0, 1]))


def test_make_mask_dataframe():
    # * Minimum Data Needed
    #   1. streak_data: Only need `streak_mask`
    #   2. min_length: This can vary

    # Set up fake data dimensions
    row_size = 1000
    col_size = 1000
    test_mask = np.zeros(shape=(row_size, col_size))

    # Initialize a new generator - set seed for reproducibility
    rng = np.random.default_rng(12345)

    # Generate Streaks
    streak_count = 50
    for _ in range(streak_count):
        (x_min,) = rng.integers(low=0, high=col_size - 1, size=1)
        (x_max,) = rng.integers(low=x_min, high=col_size - 1, size=1)
        (y,) = rng.integers(low=0, high=row_size - 1, size=1)
        streak = np.ones(shape=(x_max - x_min))
        test_mask[y, x_min:x_max] = streak

    streak_data = sd.StreakData(streak_mask=test_mask)

    # Create the mask_dataframes
    # Test various min_lengths: 50, 100, 150
    # Post filter streak counts: 40, 33, 26
    for min_l, post_filter_streak_count in zip([50, 100, 150], [40, 33, 26]):
        sd._make_mask_dataframe(streak_data=streak_data, min_length=min_l)

        assert len(streak_data.streak_df) == 48
        assert len(streak_data.filtered_streak_df) < len(streak_data.streak_df)
        assert len(streak_data.filtered_streak_df) == post_filter_streak_count


def test_make_filtered_mask(df_dataset):
    # * Minimum Data Needed
    #   1. streak_data: shape, filtered_streak_df (min_row, max_row, min_col, max_col)
    col_size = 1000
    row_size = 1000

    test_filtered_streak_df = df_dataset(shape=(row_size, col_size), df_row_num=20)

    streak_data_test = sd.StreakData(
        shape=(row_size, col_size), filtered_streak_df=test_filtered_streak_df
    )

    # run _make_filtered_mask
    sd._make_filtered_mask(streak_data=streak_data_test)
    # Make sure the filtered_streak_mask is binary
    assert np.all(np.isin(streak_data_test.filtered_streak_mask, [0, 1]))


def test_make_box_outline(df_dataset):
    # * Minimum Data Needed
    #   1. streak_data: shape, filtered_streak_df (min_row, max_row, min_col, max_col)

    col_size = 1000
    row_size = 1000

    test_filtered_streak_df = df_dataset(shape=(row_size, col_size), df_row_num=20)

    streak_data_test = sd.StreakData(
        shape=(row_size, col_size), filtered_streak_df=test_filtered_streak_df
    )

    # run _make_box_outline
    sd._make_box_outline(streak_data=streak_data_test)

    # Make sure the box outline is binary
    assert np.all(np.isin(streak_data_test.boxed_streaks, [0, 1]))


def test_make_correction_mask(df_dataset):
    # * Minimum Data Needed
    #   1. streak_data: shape, filtered_streak_df (min_row, max_row, min_col, max_col)

    col_size = 1000
    row_size = 1000

    # Create filtered_streak_df
    test_filtered_streak_df = df_dataset(shape=(row_size, col_size), df_row_num=20)

    streak_data_test = sd.StreakData(
        shape=(row_size, col_size), filtered_streak_df=test_filtered_streak_df
    )

    # run _make_correction_mask
    sd._make_correction_mask(streak_data=streak_data_test)

    # Make sure the box outline is binary
    assert np.all(np.isin(streak_data_test.corrected_streak_mask, [0, 1]))


def test_correct_streaks(df_dataset):
    # * Minimum Data Needed
    #   1. streak_data: shape, filtered_streak_df (min_row, max_row, min_col, max_col)
    #   2. Input Image: np.ndarray

    col_size = 1000
    row_size = 1000

    # Initialize a new generator - set seed for reproducibility
    rng = np.random.default_rng(12345)

    # Create filtered_streak_df
    test_filtered_streak_df = df_dataset(shape=(row_size, col_size), df_row_num=20)

    streak_data_test = sd.StreakData(
        shape=(row_size, col_size), filtered_streak_df=test_filtered_streak_df
    )

    # Create test input image
    test_input_image = rng.integers(low=0, high=16, size=(1000, 1000))

    test_corrected_image = sd._correct_streaks(
        streak_data=streak_data_test, input_image=test_input_image
    )

    # Make sure the corrected image shape matches the input image shape
    assert test_input_image.shape == test_corrected_image.shape


def test_correct_mean_alg():
    # * Minimum Data Needed
    #   1. input_image: np.ndarray
    #   2. min_row, max_row, min_col, max_col

    test_input_image = np.pad(
        np.array([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [5, 6, 7, 8, 9]]),
        pad_width=(1, 1),
        mode="edge",
    )

    min_row = 1
    max_row = 2
    min_col = 0
    max_col = 5

    # run _correct_mean_alg
    mean_streak_test = sd._correct_mean_alg(test_input_image, min_row, max_row, min_col, max_col)
    assert np.array_equal(mean_streak_test, np.array([3, 4, 5, 6, 7]))


def test_save_corrected_channels(tmp_path, streak_dataset):
    # * Minimum Data Needed
    #   1. streak_data: fov, streak_mask, streak_df, filtered_streak_mask, filtered_streak_df,
    #   boxed_streaks, corrected_streak_mask
    #   2. corrected_channels

    # Set up fake data dimensions
    chan_num = 10
    row_size = 1000
    col_size = 1000

    test_fov_data = test_utils._gen_tif_data(
        fov_number=1,
        chan_number=chan_num,
        img_shape=(row_size, col_size),
        fills=False,
        dtype=np.int16,
    )
    test_corrected_channels = test_utils.make_images_xarray(tif_data=test_fov_data)
    test_corrected_channels = test_corrected_channels[0, ...]

    # Create fake StreakData dataclass
    streak_data_test = streak_dataset(
        corrected_dir=tmp_path, fov="fov0", chan="chan0", shape=(1000, 1000)
    )

    # Save the corrected channels, do not save the sreak data
    sd.save_corrected_channels(
        streak_data=streak_data_test,
        corrected_channels=test_corrected_channels,
        data_dir=tmp_path,
        save_streak_data=False,
    )

    # Open the corrected images to make sure they were saved correctly.
    for channel in test_corrected_channels.channels.values:
        test_channel_path = Path(tmp_path, "fov0" + "-corrected", channel + ".tiff")
        saved_img = io.imread(test_channel_path)

        # Test that the correct file was saved
        assert np.array_equal(test_corrected_channels.loc[:, :, channel], saved_img)

    # Save the corrected channels, save the streak data
    sd.save_corrected_channels(
        streak_data=streak_data_test,
        corrected_channels=test_corrected_channels,
        data_dir=tmp_path,
        save_streak_data=True,
    )

    # Open the corrected images to make sure they were saved correctly.
    for channel in test_corrected_channels.channels.values:
        test_channel_path = Path(tmp_path, "fov0" + "-corrected", channel + ".tiff")
        saved_img = io.imread(test_channel_path)

        # Test that the correct file was saved
        assert np.array_equal(test_corrected_channels.loc[:, :, channel], saved_img)

    # Open the streak data csv and masks to make sure they were saved correctly.

    Read_Fn = namedtuple("Read_Operation", "read_fn, ext")
    fields = {
        "streak_mask": Read_Fn(io.imread, ".tiff"),
        "streak_df": Read_Fn(pd.read_csv, ".csv"),
        "filtered_streak_mask": Read_Fn(io.imread, ".tiff"),
        "filtered_streak_df": Read_Fn(pd.read_csv, ".csv"),
        "boxed_streaks": Read_Fn(io.imread, ".tiff"),
        "corrected_streak_mask": Read_Fn(io.imread, ".tiff"),
    }

    #  Test `streak_detection::_save`
    for field, rf in fields.items():
        if rf.ext == ".csv":
            data = rf.read_fn(
                Path(
                    tmp_path,
                    "fov0" + "-corrected",
                    f"streak_data_chan0",
                    field + rf.ext,
                ),
                index_col=0,
            )
        else:
            data = rf.read_fn(
                Path(
                    tmp_path,
                    "fov0" + "-corrected",
                    f"streak_data_chan0",
                    field + rf.ext,
                )
            )

        if type(data) is np.ndarray:
            assert np.array_equal(data, getattr(streak_data_test, field))
        if type(data) is pd.DataFrame:
            pd.testing.assert_frame_equal(data, getattr(streak_data_test, field))


def test_streak_correction():
    # * Minimum Data Needed
    #   1. fov_data: A data array containing several channels.
    #   2. streak_channel: a channel to base the streaks off of

    # Set up fake data dimensions
    chan_num = 10
    row_size = 1000
    col_size = 1000

    test_fov_data = test_utils._gen_tif_data(
        fov_number=1,
        chan_number=chan_num,
        img_shape=(row_size, col_size),
        fills=False,
        dtype=np.int16,
    )
    test_fov_data_xr = test_utils.make_images_xarray(tif_data=test_fov_data)

    visualization_fields = [
        "boxed_streaks",
        "corrected_streak_mask",
        "filtered_streak_mask",
    ]

    streak_fields = [
        "streak_mask",
        "streak_df",
        "filtered_streak_df",
    ]

    corrected_channels_test, streak_data_test = sd.streak_correction(
        fov_data=test_fov_data_xr, streak_channel="chan0", visualization_masks=False
    )
    # Assert shape
    assert corrected_channels_test.shape == test_fov_data_xr[0, ...].shape
    # Assert that the correct values in streak_data_test are present
    for vf in visualization_fields:
        assert getattr(streak_data_test, vf) is None
    for sf in streak_fields:
        assert getattr(streak_data_test, sf) is not None

    corrected_channels_test, streak_data_test = sd.streak_correction(
        fov_data=test_fov_data_xr, streak_channel="chan0", visualization_masks=True
    )
    # Assert shape
    assert corrected_channels_test.shape == test_fov_data_xr[0, ...].shape

    # Assert that the correct values in streak_data_test are present
    for vf in visualization_fields:
        assert getattr(streak_data_test, vf) is not None
    for sf in streak_fields:
        assert getattr(streak_data_test, sf) is not None
