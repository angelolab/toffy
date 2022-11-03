import pathlib
from typing import Dict, Union

import numpy as np
import skimage.io as io


def save_image(
    fname: Union[str, pathlib.Path], data: np.ndarray, compression_level: int = 8
) -> None:
    """
    A thin wrapper around `skimage.io.imsave()`.

    Args:
        fname (str): The location to save the tiff file.
        data (np.ndarray): The Numpy array to save.
        compression_level (int, optional): The compression level for skimage.io.imsave. Increasing
            `compress` increases memory consumption, decreases compression speed and moderately
            increases compression ratio. The range of compress is `[1,9]`. Defaults to 6.
    """
    # Compression Config:
    plugin_args: Dict[str, Union[str, int, Dict]] = {
        "compression": "zlib",
        "compressionargs": {"level": compression_level},
    }

    io.imsave(fname=fname, arr=data, plugin="tifffile", check_contrast=False, **plugin_args)
