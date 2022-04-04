import numpy as np
import os
from typing import Union, Optional
from pathlib import Path
from ark.utils import load_utils
from skimage import (
    filters,
    exposure,
    restoration,
    measure,
    draw,
    util,
    io,
)
from dataclasses import dataclass
import pandas as pd
from functools import partial


@dataclass()
class StreakData:
    """Contains data for correcting the streaks consisting of binary masks, dataframes with
    location and size properties, a directory for saving, and the shape / channel for mask
    generation. In addition provides a function to save any of the binary masks or dataframes.

    Args:
        shape (tuple): The shape of the image / fov.
        streak_channel (str): The specific channel name used to create the masks.
        corrected_dir (Path): The directory used to save the corrected tiffs and data in.
        streak_mask (np.ndarray): The first binary mask indicating candidate streaks.
        streak_df (pd.DataFrame): A dataframe, containing the location, area, and eccentricity
        of each streak.
        filtered_streak_mask (np.ndarray): A binary mask with out the false streaks.
        filtered_streak_df (pd.DataFrame): A subset of the `streak_df` containing location, area
        and eccentricity values of the filtered streaks.
        boxed_streaks (np.ndarray): An optional binary mask containing an outline for each
        filtered streaks.
        corrected_streak_mask (np.ndarray): An optional binary mask containing the lines used for
        correcting the streaks.
    """

    shape: tuple = None
    streak_channel: str = None
    corrected_dir: Path = None
    streak_mask: np.ndarray = None
    streak_df: pd.DataFrame = None
    filtered_streak_mask: np.ndarray = None
    filtered_streak_df: pd.DataFrame = None
    boxed_streaks: np.ndarray = None
    corrected_streak_mask: np.ndarray = None

    def __getitem__(self, item):
        return getattr(self, item)

    def save_data(self, name: Optional[str] = None, all_data: bool = True):
        """Given a field name, it saves the data as a tiff file if it's a Numpy array, and a csv
        if it is a Pandas DataFrame.

        Args:
            name (str): The field of the DataClass to be saved. Options include:
                `streak_mask`,
                `streak_df`,
                `filtered_streak_mask`,
                `filtered_streak_df`,
                `boxed_streaks`,
                `corrected_streak_mask`,
            all_data (bool): A flag for saving all the masks and dataframes. Defaults to `True`.
        """
        fields = [
            "streak_mask",
            "streak_df",
            "filtered_streak_mask",
            "filtered_streak_df",
            "boxed_streaks",
            "corrected_streak_mask",
        ]
        if all_data:
            for field in fields:
                self._save(name=field)
        elif name not in fields:
            raise ValueError(
                "Invalid field to save. Options include:\
                \n streak_mask\
                \n streak_df\
                \n filtered_streak_mask\
                \n filtered_streak_df\
                \n boxed_streaks\
                \n corrected_streak_mask"
            )
        elif name is not None:
            self._save(name)
        else:
            raise ValueError("Please enter a field name, or set `all_data` to True.")

    def _get_save_dir(self, data_dir: Path, name: str, ext: str):
        return Path(data_dir, name + f".{ext}")

    def _save(self, name: str):
        """Helper function for saving tiff binary masks and dataframes.

        Args:
            name (str): The field of the DataClass to be saved.
        """
        data_dir = Path(self.corrected_dir, f"streak_data_{self.streak_channel}")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        data = getattr(self, name)
        st = partial(self._get_save_dir, data_dir, name)

        if type(data) is np.ndarray:
            io.imsave(st("tiff"), data)
        elif type(data) is pd.DataFrame:
            data.to_csv(st("csv"))


def _make_binary_mask(
    input_image: np.ndarray,
    gaussian_sigma: float = 40,
    gamma: float = 3.80,
    gamma_gain: float = 0.10,
    log_gain: float = 1.00,
    pmin: int = 80,
    pmax: int = 95,
) -> np.ndarray:
    """Performs a series of denoiseing, filtering, and exposure adjustments to create a binary
    mask for the given input image.

    Args:
        input_image (np.ndarray): The image to perform the streak masking on.
        gaussian_sigma (float, optional): Parameter for `skimage.filters.gaussian`. Defaults to
        40.
        gamma (float, optional): Parameter for `skimage.exposure.adjust_gamma`. Defaults to 3.80.
        gamma_gain (float, optional): Parameter for `skimage.exposure.adjust_gamma`. Defaults to
        0.10.
        log_gain (float, optional): Parameter for `skimage.exposure.adjust_log`. Defaults to 1.00.
        pmin (int, optional): Lower bound for the `np.percentile` threshold, used for rescaling
        the intensity. Defaults to 80.
        pmax (int, optional): Upper bound for the `np.percentile` threshold, used for rescaling
        the intensity. Defaults to 95.

    Returns:
        np.ndarray: The binary mask containing all of the candidate strokes.
    """
    # Denoise the Image
    input_image = restoration.denoise_wavelet(
        input_image, wavelet="db2", mode="hard", rescale_sigma=True
    )
    # Rescale the intensity using percentile ranges
    pmin_v, pmax_v = np.percentile(input_image, (pmin, pmax))
    input_image = exposure.rescale_intensity(input_image, in_range=(pmin_v, pmax_v))
    # Laplace filter to get the streaks
    input_image = filters.laplace(input_image, ksize=3)
    input_image = exposure.rescale_intensity(input_image, out_range=(0, 1))
    # Smoothing
    input_image = filters.gaussian(input_image, sigma=(0, gaussian_sigma))  # (y, x)
    # Exposure Adjustments
    input_image = exposure.adjust_gamma(input_image, gamma=gamma, gain=gamma_gain)
    input_image = exposure.adjust_log(input_image, gain=log_gain, inv=True)
    input_image = exposure.rescale_intensity(input_image, out_range=(0, 1))
    # apply threshold
    binary_mask = input_image > 0.3

    return binary_mask


def _filter_mask(streak_data: StreakData, min_length: int = 50) -> None:
    """Filters the streaks. Removes incorrectly masked streaks which are too short and do not have
    a high eccentricity.

    Args:
        streak_data (StreakData): An instance of the StreakData Dataclass, holds all necessary
        data for streak correction.
        min_length (int): The lower threshold for filtering streaks in pixels. Defaults to 50.
    """
    # Label all the candidate streaks
    labeled_streaks = measure.label(
        streak_data.streak_mask, connectivity=2, return_num=False
    )

    # Gather properties of all the candidate streaks using regionprops.
    region_properties = measure.regionprops_table(
        label_image=labeled_streaks,
        cache=True,
        properties=[
            "label",
            "bbox",
            "eccentricity",
            "area",
        ],
    )

    # Convert dictionary of region properties to DataFrame.
    streak_data.streak_df = pd.DataFrame(region_properties)

    # Rename the bounding box columns.
    streak_data.streak_df.rename(
        {
            "bbox-0": "min_row",
            "bbox-1": "min_col",
            "bbox-2": "max_row",
            "bbox-3": "max_col",
        },
        axis="columns",
    )
    # Give the index column a name.
    streak_data.streak_df.index.names = ["index"]

    # Filter out eccentricities that are less than 0.99999 (only keep straight lines)
    # Filter out small areas (small lines)
    eccentricity_value = 0.99999
    area_value = min_length
    streak_data.filtered_streak_df = streak_data.streak_df.query(
        f"eccentricity > {eccentricity_value} and area > {area_value}"
    )


def _make_filtered_mask(streak_data: StreakData) -> None:
    """Creates the binary streak mask using the filtered streak DataFrame. These are the streaks
    which will get corrected.

    Args:
        streak_data (StreakData): An instance of the StreakData Dataclass, holds all necessary
        data for streak correction.
    """
    streak_data.filtered_streak_mask = np.zeros(shape=streak_data.shape, dtype=np.int8)
    for region in streak_data.filtered_streak_df.itertuples():
        streak_data.filtered_streak_mask[
            region.min_row : region.max_row, region.min_col : region.max_col
        ] = 1


def _make_box_outline(streak_data: StreakData) -> None:
    """Creates a box outline for each binary streak using the filtered streak DataFrame. Outlines
    the streaks that will get corrected. Primarily for visualization / debugging purposes.

    Args:
        streak_data (StreakData): An instance of the StreakData Dataclass, holds all necessary
        data for streak correction.
    """
    streak_data.boxed_streaks = np.zeros(shape=streak_data.shape, dtype=np.int8)
    for region in streak_data.filtered_streak_df.itertuples():
        y, x = draw.rectangle_perimeter(
            start=(region.min_row, region.min_col),
            end=(region.max_row - 1, region.max_col - 1),
            clip=True,
            shape=streak_data.shape,
        )
        streak_data.boxed_streaks[y, x] = 1


def _make_correction_mask(streak_data: StreakData) -> None:
    """Creates the correction mask for each binary streak using the filtered streak DataFrame.
    Marks pixels which will be used for the correction method.

    Args:
        streak_data (StreakData): An instance of the StreakData Dataclass, holds all necessary
        data for streak correction.
    """
    padded_image = np.pad(
        np.zeros(shape=streak_data.shape, dtype=np.int8), pad_width=(1, 1), mode="edge"
    )

    for region in streak_data.filtered_streak_df.itertuples():
        padded_image[
            region.min_row, region.min_col + 1 : region.max_col + 1
        ] = np.ones(shape=(region.max_col - region.min_col))
        padded_image[
            region.max_row + 1, region.min_col + 1 : region.max_col + 1
        ] = np.ones(shape=(region.max_col - region.min_col))

    streak_data.corrected_streak_mask = util.crop(padded_image, crop_width=(1, 1))


def _correct_streaks(streak_data: StreakData, input_image: np.ndarray) -> np.ndarray:
    """Corrects the streaks for the input image. Uses masks in the streak_data Dataclass.
    Performs the correction by averaging the pixels above and below the streak.

    Args:
        streak_data (StreakData): An instance of the StreakData Dataclass, holds all necessary
        data for streak correction.
        input_image (np.ndarray): The input image (Numpy array representation of the tiff file)
        to be used for streak detection.

    Returns:
        np.ndarray: The corrected image.
    """
    # Pad the image for edge cases.
    padded_image = np.pad(input_image.copy(), pad_width=(1, 1), mode="edge")
    corrected_image = padded_image.copy()
    # Correct each streak
    for region in streak_data.filtered_streak_df.itertuples():
        corrected_image[
            region.max_row, region.min_col : region.max_col
        ] = _mean_correction(
            padded_image,
            region.min_row,
            region.max_row,
            region.min_col,
            region.max_col,
        )
    # Crop and return the 'unpadded' image.
    return util.crop(corrected_image, crop_width=(1, 1), copy=True)


def _mean_correction(
    input_image: np.ndarray, min_row: int, max_row: int, min_col: int, max_col: int
) -> np.ndarray:
    """Performs streak-wise correction by: setting the value of each pixel in the streak to the
    mean of pixel above and below it.

    Args:
        input_image (np.ndarray): The input image (Numpy array representation of the tiff file) to
        be used for streak correction.
        min_row (int): The minimum row index of the streak. The y location where the streak
        starts.
        max_row (int): The maximum row index of the streak. The y location where the streak ends.
        min_col (int): The minimum column index of the streak. The x location where the streak
        starts.
        max_col (int): The maximum column index of the streak. The x location where the streak
        ends.

    Returns:
        np.ndarray: Returns the corrected streak.
    """
    streak_corrected: np.ndarray = np.mean(
        [
            # Row above
            input_image[min_row, min_col + 1 : max_col + 1],
            # Row below
            input_image[max_row + 1, min_col + 1 : max_col + 1],
        ],
        axis=0,
        dtype=np.int8,
    )

    return streak_corrected


def streak_correction(
    fov_dir: Union[str, Path],
    image_name: str = "Noodle",
    mask_data: bool = False,
) -> Optional[StreakData]:
    """Takes a `fov` directory and a user specified image for streak detection. Once all the
    streaks have been detected on that image, they are corrected via an averaging method. The
    function can also return a DataClass containing various binary masks and dataframes which were
    used for filtering and correction when prompted to.

    Args:
        fov_dir (Union[str, Path]): A directory containing the fov and all it's channels for
        correction.
        image_name (str): The name of the image used for identifying and removing the streaks.
        Defaults to "Noodle".
        mask_data (bool): If `True`, returns a DataClass consisting of binary masks and dataframes
        for the detected streaks. Defaults to "False"

    Returns:
        Optional[StreakData]: A DataClass holding all necessary data for streak detection and
        correction as well as masks useful for debugging and visualization.
    """

    # Initialize the streak DataClass
    streak_data = StreakData(streak_channel=image_name)

    # Open fov directory containing all the tiff files
    fov_data = load_utils.load_imgs_from_dir(data_dir=fov_dir)
    
    #  Open the image for mask generation.
    with fov_data.sel(fovs=image_name) as data:
        # Reshape it from (x,x,1) to (x,x)
        input_image = data.values.reshape(2048,-1)
        streak_data.shape = input_image.shape
        # Create and filter the binary masks
        streak_data.streak_mask = _make_binary_mask(input_image = input_image)
        _filter_mask(streak_data=streak_data, min_length=50)
    
    # Get the file names.
    channel_fn = fov_data.fovs.values.tolist()
    
    # Initialize the corrected images.
    corrected_images = {
        fn: np.zeros(shape=streak_data.shape, dtype=np.int8) for fn in channel_fn
    }

    for data in fov_data:
        name = data.fovs.values
        input_image = data.values.reshape(2048, -1)
        corrected_images[name] = _correct_streaks(
            streak_data=streak_data, input_image=input_image
        )

    # Create the directory to store the corrected tiffs
    streak_data.corrected_dir = Path(fov_dir, "corrected")
    if not os.path.exists(streak_data.corrected_dir):
        os.makedirs(streak_data.corrected_dir)

    # Save the corrected tiffs
    for fn, img in corrected_images.items():
        fp = Path(streak_data.corrected_dir, fn + ".tiff")
        io.imsave(fp, img)

    # Add mask information and return it
    if mask_data:
        _make_box_outline(streak_data=streak_data)
        _make_correction_mask(streak_data=streak_data)
        _make_filtered_mask(streak_data=streak_data)
        return streak_data
