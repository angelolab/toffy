import os
from dataclasses import dataclass, field
import inspect

import pandas as pd
import xarray as xr

from mibi_bin_tools.bin_files import extract_bin_files, _write_out

from toffy.qc_comp import compute_qc_metrics_direct


@dataclass
class FovCallbacks:
    run_folder: str
    point_name: str
    __panel: pd.DataFrame = field(default=None, init=False)
    __fov_data: xr.DataArray = field(default=None, init=False)

    def _generate_fov_data(self, panel: pd.DataFrame, intensities=False, time_res=0.0005,
                           **kwargs):
        """Extracts data from bin files using the given panel

        The data and the panel are then cached members of the FovCallbacks object

        Args:
            panel (pd.DataFrame):
                Panel used for extraction
            intensities (bool | List[str]):
                Intensities argument for `mibi_bin_tools.bin_files.extract_bin_files`
            time_res (float):
                Time resolution argument for `mibi_bin_tool.bin_files.extract_bin_files`
            **kwargs (dict):
                Unused kwargs for other functions
        """
        self.__fov_data = extract_bin_files(
            data_dir=self.run_folder,
            out_dir=None,
            include_fovs=[self.point_name],
            panel=panel,
            intensities=intensities,
            time_res=time_res
        )

        self.__panel = panel

    def extract_tiffs(self, tiff_out_dir: str, panel: pd.DataFrame, **kwargs):
        """Extract tiffs into provided directory, using given panel

        Args:
            tiff_out_dir (str):
                Path where tiffs are written
            panel (pd.DataFrame):
                Target mass integration ranges
            **kwargs (dict):
                Additional arguments for `mibi_bin_tools.bin_files.extract_bin_files`.
                Accepted kwargs are

             - intensities
             - time_res
        """
        if not os.path.exists(tiff_out_dir):
            os.makedirs(tiff_out_dir)

        if self.__fov_data is None:
            self._generate_fov_data(panel, **kwargs)

        intensities = kwargs.get('intensities', False)
        _write_out(
            img_data=self.__fov_data[0, :, :, :, :].values,
            out_dir=tiff_out_dir,
            fov_name=self.point_name,
            targets=list(self.__fov_data.channel.values),
            intensities=intensities
        )

    def generate_qc(self, qc_out_dir: str, panel: pd.DataFrame = None, **kwargs):
        """Genereates qc metrics from given panel, and saves output to provided directory

        Args:
            qc_out_dir (str):
                Path where qc_metrics are written
            panel (pd.DataFrame):
                Target mass integration ranges
            **kwargs (dict):
                Additional arguments for `toffy.qc_comp.compute_qc_metrics`. Accepted kwargs are:

             - gaussian_blur
             - blur_factor
        """
        if not os.path.exists(qc_out_dir):
            os.makedirs(qc_out_dir)

        if self.__fov_data is None:
            if panel is None:
                raise ValueError('Must provide panel if fov data is not already generated...')
            self._generate_fov_data(panel, **kwargs)

        metric_data = compute_qc_metrics_direct(
            image_data=self.__fov_data,
            fov_name=self.point_name,
            gaussian_blur=kwargs.get('gaussian_blur', False),
            blur_factor=kwargs.get('blur_factor', 1)
        )

        for metric_name, data in metric_data.items():
            data.to_csv(os.path.join(qc_out_dir, metric_name), index=False)


def build_fov_callback(*args, **kwargs):

    # validate user callback settings
    methods = [attr for attr in dir(FovCallbacks) if attr[0] != '_']
    for arg in args:
        if arg not in methods:
            raise ValueError(
                f'{arg} is not a valid FovCallbacks member\n'
                f'Accepted callbacks are {methods}'
            )
        argnames = inspect.getfullargspec(getattr(FovCallbacks, arg))[0]
        for argname in argnames:
            if argname not in kwargs and argname != 'self':
                raise ValueError(
                    f'Missing necessary keyword argument, {argname} for callback function {arg}...'
                )

    # construct actual callback
    def fov_callback(run_folder: str, point_name: str):
        callback_obj = FovCallbacks(run_folder, point_name)
        for arg in args:
            if cb := getattr(callback_obj, arg, None):
                cb(**kwargs)
            else:
                # unreachable...
                raise ValueError(
                    f'Could not locate attribute {arg} in FovCallback object'
                )

    return fov_callback
