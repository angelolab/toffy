import os
from typing import Callable

import pandas as pd

from mibi_bin_tools.bin_files import extract_bin_files

from toffy.qc_comp import compute_qc_metrics


def build_extract_callback(out_dir: str, panel: pd.DataFrame,
                           **kwargs) -> Callable[[str, str], None]:
    """Generates extraction callback for given panel + parameters

    Args:
        out_dir (str):
            Path where tiffs are written
        panel (pd.DataFrame):
            Target mass integration ranges
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

    def extract_callback(run_folder: str, point_name: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        extract_bin_files(run_folder, out_dir, [point_name], panel, **kwargs)

    return extract_callback


def build_qc_callback(out_dir: str, panel: pd.DataFrame, **kwargs) -> Callable[[str, str], None]:
    """Generates qc callback for given panel + parameters

    Args:
        out_dir (str):
            Path where qc metrics are written
        panel (pd.DataFrame):
            Target mass integration ranges
        **kwargs (dict):
            Additional arguments for `toffy.qc_comp.compute_qc_metrics`.  Accepted kwargs are:

         - gaussian_blur
         - blur_factor

    Returns:
        Callable[[str, str], None]:
            Callback for fov watcher
    """

    if isinstance(panel, tuple):
        raise TypeError('Global unit mass integration is no longer support. Please provide panel '
                        'as a pandas DataFrame...')

    kwargs['save_csv'] = False

    def qc_callback(run_folder: str, point_name: str):
        metric_data = compute_qc_metrics(run_folder, point_name, None, panel, **kwargs)
        for metric_name, data in metric_data.items():
            data.to_csv(os.path.join(out_dir, metric_name), index=False)

    return qc_callback
