import os
from typing import Union, Tuple, Callable

import pandas as pd

from mibi_bin_tools.bin_files import extract_bin_files

from toffy.qc_comp import compute_qc_metrics


def build_extract_and_compute_qc_callback(panel: Union[Tuple[float, float], pd.DataFrame],
                                          extraction_dir_name: str = 'extracted',
                                          **kwargs) -> Callable[[str, str, str], None]:
    """Generates extraction and qc metric computation callback for given panel + parameters

    Args:
        panel (tuple | pd.DataFrame):
            Target mass integration ranges
        extraction_dir_name (str):
            Subdirectory to place extracted TIFs into
        **kwargs (dict):
            Additional arguments for `mibi_bin_tools.bin_files.extract_bin_files` and
            `toffy.qc_comp.compute_qc_metrics`.  Accepted kwargs are:

         - intensities
         - time_res
         - channels
         - gaussian_blur
         - blur_factor
         - dtype

    Returns:
        Callable[[str, str, str], None]:
            Callback for fov watcher
    """

    intensities = kwargs.pop('intensities', False)
    time_res = kwargs.pop('time_res', 0.005)

    def extract_and_compute_qc_callback(run_folder: str, point_name: str, out_dir: str):
        extraction_dir = os.path.join(out_dir, extraction_dir_name)
        if not os.path.exists(extraction_dir):
            os.makedirs(extraction_dir)

        extract_bin_files(run_folder, extraction_dir, [point_name], panel,
                          intensities, time_res)

        qc_data = compute_qc_metrics(extraction_dir, img_sub_folder=None, fovs=[point_name],
                                     **kwargs)
        for name, metric_data in qc_data.items():
            metric_data.to_csv(os.path.join(out_dir, f'{name}.csv'))

    return extract_and_compute_qc_callback
