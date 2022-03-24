import os
from typing import Callable

import pandas as pd

from mibi_bin_tools.bin_files import extract_bin_files

from toffy.qc_comp import compute_qc_metrics


def build_extract_callback(panel: pd.DataFrame, extraction_dir_name: str = 'extracted',
                           **kwargs) -> Callable[[str, str, str], None]:
    """Generates extraction callback for given panel + parameters

    Args:
        panel (pd.DataFrame):
            Target mass integration ranges
        extraction_dir_name (str):
            Subdirectory to place extracted TIFs into
        **kwargs (dict):
            Additional arguments for `mibi_bin_tools.bin_files.extract_bin_files`.
            Accepted kwargs are:

         - intensities
         - time_res

    Returns:
        Callable[[str, str, str], None]:
            Callback for fov watcher
    """

    if isinstance(panel, tuple):
        raise TypeError('Global unit mass integration is no longer support. Please provide panel '
                        'as a pandas DataFrame...')

    def extract_callback(run_folder: str, point_name: str, out_dir: str):

        extraction_dir = os.path.join(out_dir, extraction_dir_name)
        if not os.path.exists(extraction_dir):
            os.makedirs(extraction_dir)

        extract_bin_files(run_folder, extraction_dir, [point_name], panel, **kwargs)

    return extract_callback


def build_qc_callback(panel: pd.DataFrame, **kwargs) -> Callable[[str, str, str], None]:
    """Generates qc callback for given panel + parameters

    Args:
        panel (pd.DataFrame):
            Target mass integration ranges
        **kwargs (dict):
            Additional arguments for `toffy.qc_comp.compute_qc_metrics`.  Accepted kwargs are:

         - gaussian_blur
         - blur_factor

    Returns:
        Callable[[str, str, str], None]:
            Callback for fov watcher
    """

    if isinstance(panel, tuple):
        raise TypeError('Global unit mass integration is no longer support. Please provide panel '
                        'as a pandas DataFrame...')

    def qc_callback(run_folder: str, point_name: str, out_dir: str):
        compute_qc_metrics(run_folder, point_name, None, panel, **kwargs)

    return qc_callback
