from typing import Tuple, Callable
import pytest
from pathlib import Path
from toffy import streak_detection as sd
import numpy as np
import pandas as pd
from skimage import io
from collections import namedtuple
from ark.utils import test_utils


@pytest.fixture(scope="function")
def streak_dataset() -> Callable:
    """A wrapper for creating a StreakData DataClass for testing purposes.

    Returns:
        function: Returns a function which generates an instance of the StreakData DataClass.
    """
    # Return a function which generates an instance of the StreakData DataClass
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
def df_dataset() -> Callable:
    """A wrapper for creating a dataframe for testing purposes.

    Returns:
        function: Returns a function which generates a DataFrame with location data
        of streaks.
    """
    # Return a function which generates an instance of a dataframe with streak location data.
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
        _test_df = pd.DataFrame(
            np.stack([min_row, max_row, min_col, max_col], axis=-1),
            columns=["min_row", "max_row", "min_col", "max_col"],
        )
        return _test_df

    return test_df_generator


@pytest.fixture(scope="function")
def create_image() -> Callable:
    """A wrapper for generating streaked image for testing purposes.

    Returns:
        function: Returns a function which creates an image with streaks.
    """
    # Returns a function which generates a test image with streak location data
    def image_generator(shape: Tuple[int, int], streak_count: int):
        # Set up fake data dimensions
        row_size = shape[0]
        col_size = shape[1]
        _test_image = np.zeros(shape=(row_size, col_size))

        # Initialize a new generator - set seed for reproducibility
        rng = np.random.default_rng(12345)

        # Generate Streaks
        for _ in range(streak_count):
            (x_min,) = rng.integers(low=0, high=col_size - 1, size=1)
            (x_max,) = rng.integers(low=x_min, high=col_size - 1, size=1)
            (y,) = rng.integers(low=0, high=row_size - 1, size=1)
            streak = np.ones(shape=(x_max - x_min))
            _test_image[y, x_min:x_max] = streak
        return _test_image

    return image_generator


def test_get_save_dir(tmp_path: Path):
    """Tests the helper function which generates a file path fora tiff file or a csv to be saved.
    Tests both in a temporary directory.

    Args:
        tmp_path (Path): A fixture which will provide a temporary directory unique to the test
        invocation.
    """
    data_dir: Path = Path(tmp_path)
    name: str = "streak_data_test"
    ext_csv: str = "csv"
    ext_tiff: str = "tiff"

    # Assert that the path to saving the csv file is correct
    csv_save_dir: Path = sd._get_save_dir(data_dir=data_dir, name=name, ext=ext_csv)
    assert csv_save_dir == Path(tmp_path / "streak_data_test.csv")

    # Assert that the path to saving the tiff file is correct
    tiff_save_dir: Path = sd._get_save_dir(data_dir=data_dir, name=name, ext=ext_tiff)
    assert tiff_save_dir == Path(tmp_path / "streak_data_test.tiff")


def test_save_streak_data(tmp_path: Path, streak_dataset: Callable):
    """Tests that the data in an instance of StreakData saves correctly with the helper function.

    Args:
        tmp_path (Path): A fixture which will provide a temporary directory unique to the test
        invocation.
        streak_dataset (Callable): A wrapper for creating a StreakData DataClass for testing
        purposes.
    """
    streak_data_test = streak_dataset(
        corrected_dir=tmp_path, fov="fov0", chan="chan0", shape=(20, 20)
    )

    # Creates a dictionary where the key is the field to be saved / tested and the value is a
    # namedtuple containing the function which can read the field, and the file extension
    Read_Fn = namedtuple("Read_Operation", "read_fn, ext")
    fields = {
        "streak_mask": Read_Fn(io.imread, ".tiff"),
        "streak_df": Read_Fn(pd.read_csv, ".csv"),
        "filtered_streak_mask": Read_Fn(io.imread, ".tiff"),
        "filtered_streak_df": Read_Fn(pd.read_csv, ".csv"),
        "boxed_streaks": Read_Fn(io.imread, ".tiff"),
        "corrected_streak_mask": Read_Fn(io.imread, ".tiff"),
    }

    for field, rf in fields.items():
        sd._save_streak_data(streak_data=streak_data_test, name=field)
        # Read the field
        if rf.ext == ".csv":
            data = rf.read_fn(Path(tmp_path, f"streak_data_chan0", field + rf.ext), index_col=0)
        else:
            data = rf.read_fn(Path(tmp_path, f"streak_data_chan0", field + rf.ext))

        # Assert that the correct numpy array was saved as a tiff.
        if type(data) is np.ndarray:
            assert np.array_equal(data, getattr(streak_data_test, field))
        # Assert that the correct DataFrame was saved as a csv.
        if type(data) is pd.DataFrame:
            pd.testing.assert_frame_equal(data, getattr(streak_data_test, field))


def test_save_streak_masks(tmp_path: Path, streak_dataset: Callable):
    """Tests that the data in an instance of StreakData saves correctly with the save_streak_masks
    function.

    Args:
        tmp_path (Path): A fixture which will provide a temporary directory unique to the test
        invocation.
        streak_dataset (Callable): A wrapper for creating a StreakData DataClass for testing
        purposes.
    """
    streak_data_test = streak_dataset(
        corrected_dir=tmp_path, fov="fov0", chan="chan0", shape=(20, 20)
    )

    sd._save_streak_masks(streak_data=streak_data_test)

    # Creates a dictionary where the key is the field to be saved / tested and the value is a
    # namedtuple containing the function which can read the field, and the file extension
    Read_Fn = namedtuple("Read_Operation", "read_fn, ext")
    fields = {
        "streak_mask": Read_Fn(io.imread, ".tiff"),
        "streak_df": Read_Fn(pd.read_csv, ".csv"),
        "filtered_streak_mask": Read_Fn(io.imread, ".tiff"),
        "filtered_streak_df": Read_Fn(pd.read_csv, ".csv"),
        "boxed_streaks": Read_Fn(io.imread, ".tiff"),
        "corrected_streak_mask": Read_Fn(io.imread, ".tiff"),
    }

    for field, rf in fields.items():
        if rf.ext == ".csv":
            data = rf.read_fn(Path(tmp_path, f"streak_data_chan0", field + rf.ext), index_col=0)
        else:
            data = rf.read_fn(Path(tmp_path, f"streak_data_chan0", field + rf.ext))

        # Assert that the correct numpy array was saved as a tiff.
        if type(data) is np.ndarray:
            assert np.array_equal(data, getattr(streak_data_test, field))
        # Assert that the correct DataFrame was saved as a csv.
        if type(data) is pd.DataFrame:
            pd.testing.assert_frame_equal(data, getattr(streak_data_test, field))


def test_make_binary_mask(create_image: Callable):
    """Tests that the binary mask is created, and is correct.

    Args:
        create_image (Callable): A wrapper for generating streaked image for testing purposes.
    """
    # Set up fake data dimensions
    row_size = 200
    col_size = 200
    input_image_test = create_image(shape=(row_size, col_size), streak_count=5)

    binary_mask = sd._make_binary_mask(input_image=input_image_test)
    # Assert that the binary mask contains only 0 or 1.
    assert np.all(np.isin(binary_mask, [0, 1]))

    # Get the overlapping streaks from the binary mask and the input image
    detected_x, detected_y = np.equal(binary_mask, input_image_test).nonzero()
    detected_streak_coords = list(zip(detected_x, detected_y))
    input_x, input_y = input_image_test.nonzero()
    input_image_streak_coords = list(zip(input_x, input_y))
    shared_pts = [pt for pt in input_image_streak_coords if pt in detected_streak_coords]

    # Assert that the correct pixels are marked with the binary mask.
    assert len(shared_pts) == 242


def test_make_mask_dataframe(create_image):
    """Tests that the streak dataframe is correctly constructed, with the correct locations,
    and that filtering can be done with `min_length`.

    Args:
        create_image (Callable): A wrapper for generating streaked image for testing purposes.
    """
    # Set up fake data dimensions
    row_size = 1000
    col_size = 1000
    input_test_mask = create_image(shape=(row_size, col_size), streak_count=50)

    streak_data = sd.StreakData(streak_mask=input_test_mask)

    # Create the mask_dataframes
    # Test various min_lengths: 50, 100, 150
    # Post filter streak counts: 40, 33, 26
    for min_l, post_filter_streak_count in zip([50, 100, 150], [40, 33, 26]):
        sd._make_mask_dataframe(streak_data=streak_data, min_length=min_l)

        assert len(streak_data.streak_df) == 48
        assert len(streak_data.filtered_streak_df) < len(streak_data.streak_df)
        assert len(streak_data.filtered_streak_df) == post_filter_streak_count


def test_make_filtered_mask(df_dataset: Callable):
    """Tests that the filtered mask is made correctly from a test dataframe.

    Args:
        df_dataset (Callable): A wrapper for creating a dataframe for testing purposes.
    """
    col_size = 20
    row_size = 20

    test_filtered_streak_df = df_dataset(shape=(row_size, col_size), df_row_num=20)

    streak_data_test = sd.StreakData(
        shape=(row_size, col_size), filtered_streak_df=test_filtered_streak_df
    )

    sd._make_filtered_mask(streak_data=streak_data_test)
    # Make sure the filtered_streak_mask is binary
    assert np.all(np.isin(streak_data_test.filtered_streak_mask, [0, 1]))


def test_make_box_outline(df_dataset: Callable):
    """Tests that the boxed outline is made correctly from a test dataframe.

    Args:
        df_dataset (Callable): A wrapper for creating a dataframe for testing purposes.
    """
    col_size = 20
    row_size = 20

    test_filtered_streak_df = df_dataset(shape=(row_size, col_size), df_row_num=20)

    streak_data_test = sd.StreakData(
        shape=(row_size, col_size), filtered_streak_df=test_filtered_streak_df
    )

    sd._make_box_outline(streak_data=streak_data_test)

    # Make sure the box outline is binary
    assert np.all(np.isin(streak_data_test.boxed_streaks, [0, 1]))


def test_make_correction_mask(df_dataset: Callable):
    """Tests that the correction mask is made correctly from a test dataframe.

    Args:
        df_dataset (Callable): A wrapper for creating a dataframe for testing purposes.
    """
    col_size = 20
    row_size = 20

    # Create filtered_streak_df
    test_filtered_streak_df = df_dataset(shape=(row_size, col_size), df_row_num=20)

    streak_data_test = sd.StreakData(
        shape=(row_size, col_size), filtered_streak_df=test_filtered_streak_df
    )

    sd._make_correction_mask(streak_data=streak_data_test)

    # Make sure the box outline is binary
    assert np.all(np.isin(streak_data_test.corrected_streak_mask, [0, 1]))


def test_correct_streaks(df_dataset: Callable):
    """Test that the corrected image is generated properly.

    Args:
        df_dataset (Callable): A wrapper for creating a dataframe for testing purposes.
    """
    col_size = 20
    row_size = 20

    # Initialize a new generator - set seed for reproducibility
    rng = np.random.default_rng(12345)

    # Create filtered_streak_df
    test_filtered_streak_df = df_dataset(shape=(row_size, col_size), df_row_num=20)

    streak_data_test = sd.StreakData(
        shape=(row_size, col_size), filtered_streak_df=test_filtered_streak_df
    )

    # Create test input image
    test_input_image = rng.integers(low=0, high=16, size=(20, 20))

    test_corrected_image = sd._correct_streaks(
        streak_data=streak_data_test, input_image=test_input_image
    )

    # Make sure the corrected image shape matches the input image shape
    assert test_input_image.shape == test_corrected_image.shape


def test_correct_mean_alg():
    """Tests that the mean algorithm for fixing the streaks behaves properly."""
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
    """Tests that the channels (images) are saved correctly.

    Args:
        tmp_path (Path): A fixture which will provide a temporary directory unique to the test
        invocation.
        streak_dataset (Callable): A wrapper for creating a StreakData DataClass for testing
        purposes.
    """
    # Set up fake data dimensions
    chan_num = 10
    row_size = 20
    col_size = 20

    # Generate fake tiff data with ark.test_utils
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
        corrected_dir=tmp_path, fov="fov0", chan="chan0", shape=(row_size, col_size)
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

        # Assert that the correct file was saved
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

        # Assert that the correct file was saved
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
            # Assert that the proper tiff is loaded.
            assert np.array_equal(data, getattr(streak_data_test, field))
        if type(data) is pd.DataFrame:
            # Assert that the proper csv is loaded.
            pd.testing.assert_frame_equal(data, getattr(streak_data_test, field))


def test_streak_correction():
    """Tests that the streaks are corrected properly."""
    # Set up fake data dimensions
    chan_num = 10
    row_size = 20
    col_size = 20

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
