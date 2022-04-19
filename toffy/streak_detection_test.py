from dataclasses import dataclass
from pathlib import Path
from typing import List
from attr import field
import numpy as np
import xarray as xr
from ark.utils import load_utils
import pytest
import tempfile
from toffy import streak_detection as sd
from collections import Counter


@dataclass
class Fov:
    fov_name: str
    fov_da: xr.DataArray
    corrected_channels: field(default=None)
    streak_data: sd.StreakData = field(default=None)


@pytest.fixture(scope="class")
def get_fovs(request):
    # Load the data, use class scope to cache the fovs
    # Once the class tests are done, it's removed from memory
    data_dir = Path("toffy/data/streak/")
    fovs = [fov.stem for fov in Path(data_dir).glob("*//") if "corrected" not in fov.stem]
    fovs_dc = [
        Fov(
            fov_name=fov_name,
            fov_da=load_utils.load_imgs_from_tree(
                data_dir=data_dir, fovs=[fov_name], dtype=np.int32
            ),
            corrected_channels=None,
            streak_data=None,
        )
        for fov_name in fovs
    ]
    request.cls.fovs = fovs_dc
    request.cls.data_dir = data_dir


@pytest.mark.skip(reason="Cannot test with real data, need to utilize synthetic data")
@pytest.mark.usefixtures("get_fovs")
class TestStreakDetection:
    def test_streak_detection(self):
        # Test to make sure all the fovs are loaded correctly
        assert len(self.fovs) == 3

        # Test the end-to-end streak correction pipeline function
        #   `streak_detection::streak_correction`
        # Make sure the input shape is equal to the output shape
        # Add the corrected_channels and the streak_data to the Fov dataclass
        for idx, fov in enumerate(self.fovs):
            corrected_channels, streak_data = sd.streak_correction(
                fov_data=fov.fov_da, streak_channel="Noodle", visualization_masks=True
            )
            assert corrected_channels.shape == fov.fov_da[0, ...].shape
            self.fovs[idx].corrected_channels = corrected_channels
            self.fovs[idx].streak_data = streak_data

    def test_save_corrected_channels(self):
        with tempfile.TemporaryDirectory() as save_dir:

            save_dir_path = Path(save_dir)

            for fov in self.fovs:
                # Save Streak Masks
                sd.save_corrected_channels(
                    streak_data=fov.streak_data,
                    corrected_channels=fov.corrected_channels,
                    data_dir=save_dir,
                    save_streak_data=True,
                )
            # Gather all the saved files
            fovs = [fov.stem for fov in save_dir_path.glob("*//") if "corrected" in fov.stem]

            # Check that the names in the corrected files are the same as the original files

            for corrected_fov, fov in zip(fovs, self.fovs):
                corrected_tiff_names = [
                    cor_tiff.stem
                    for cor_tiff in Path(save_dir_path / corrected_fov).glob("*.tiff")
                ]

                # Get initial file names
                initial_tiff_names = [
                    init_tiff.stem
                    for init_tiff in Path(self.data_dir / fov.fov_name).glob("*.tiff")
                ]

                # Test that the set of initial tiffs map 1-to-1 to set of corrected tiffs
                assert Counter(initial_tiff_names) == Counter(corrected_tiff_names)


# ? Construct different sets of fake data (noisy, and clean variants)
# ? Modify to work with `parameterize_with_cases`
# TODO: All channels of fake data must contain the same streaks. Add noise to each channel.
# TODO: Add multiple streaks.
# TODO: Hardcode the streak tests first, request for review
@pytest.fixture(scope="class", params=[2, 4, 10])
def fake_data(request):
    # Set up fake data dimensions
    num_images = 10
    row_size = 1000
    col_size = 1000
    fake_data = np.zeros(shape=(row_size, col_size, num_images), dtype=np.uint8)

    # Initialize a new generator
    rng = np.random.default_rng(12345)

    # Generate Streaks
    for _ in range(request.param):
        (x_min,) = rng.integers(low=0, high=col_size - 50, size=1)
        (x_max,) = rng.integers(low=x_min + 50, high=col_size - 1, size=1)
        (y,) = rng.integers(low=0, high=row_size - 1, size=1)
        fake_streak = np.ones(shape=(x_max - x_min, num_images))
        fake_data[y, x_min:x_max, :] = fake_streak

    # for fd in range(num_images):

    # Create fake DataArray
    fake_data_da = xr.DataArray(
        data=fake_data,
        coords=[range(row_size), range(col_size), range(num_images)],
        dims=["rows", "cols", "channels"],
    )
    # Create fake StreakData dataclass
    # Use channel 0 as the streak detection channel.
    streak_data: sd.StreakData = sd.StreakData(
        shape=(row_size, col_size), fov="fake_fov", streak_channel=0
    )

    # Add the Fake Fov to the request object
    request.cls.fake_fov = Fov(
        fov_name="fake_fov", fov_da=fake_data_da, corrected_channels=None, streak_data=streak_data
    )


@pytest.mark.usefixtures("fake_data")
class TestHelperFunctions:
    def test_make_binary_mask(self):

        for channel in self.fake_fov.fov_da.channels.values:
            fake_img_bin_mask: np.ndarray = sd._make_binary_mask(
                input_image=self.fake_fov.fov_da.loc[:, :, channel].values
            )
            # Make sure the masks only contain values: `0`, `1`
            assert np.all(np.isin(fake_img_bin_mask, [0, 1]))

            # Make sure the shape is the same
            assert fake_img_bin_mask.shape == self.fake_fov.streak_data.shape

    @pytest.mark.parametrize("min_length", [50])
    def test_make_mask_dataframe(self, min_length):

        fake_img_bin_mask = sd._make_binary_mask(
            input_image=self.fake_fov.fov_da.loc[:, :, self.fake_fov.streak_data.streak_channel]
        )
        self.fake_fov.streak_data.streak_mask = fake_img_bin_mask

        sd._make_mask_dataframe(streak_data=self.fake_fov.streak_data, min_length=min_length)

        assert len(self.fake_fov.streak_data.streak_df) >= len(
            self.fake_fov.streak_data.filtered_streak_df
        )

    def test_make_filtered_mask(self):
        sd._make_filtered_mask(self.fake_fov.streak_data)

        # Make sure the filtered mask only contains values: `0`, `1`
        assert np.all(np.isin(self.fake_fov.streak_data.filtered_streak_mask, [0, 1]))

    def test_make_box_outline(self):
        sd._make_box_outline(self.fake_fov.streak_data)

        # Make sure the filtered mask only contains values: `0`, `1`
        assert np.all(np.isin(self.fake_fov.streak_data.boxed_streaks))
