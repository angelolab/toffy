import inspect
import os
import warnings
from dataclasses import dataclass, field
from typing import Iterable

# prevent memory leaking from creating plots that are never shown
import matplotlib
import pandas as pd
import xarray as xr
from alpineer import io_utils, misc_utils
from mibi_bin_tools.bin_files import _write_out, extract_bin_files
from mibi_bin_tools.type_utils import any_true

from toffy.bin_extraction import incomplete_fov_check
from toffy.image_stitching import stitch_images
from toffy.json_utils import missing_fov_check
from toffy.mph_comp import combine_mph_metrics, compute_mph_metrics, visualize_mph
from toffy.normalize import write_mph_per_mass
from toffy.panel_utils import modify_panel_ranges
from toffy.qc_comp import combine_qc_metrics, compute_qc_metrics_direct
from toffy.qc_metrics_plots import visualize_qc_metrics
from toffy.settings import QC_COLUMNS

matplotlib.use("Agg")

RUN_PREREQUISITES = {
    "plot_qc_metrics": set(["generate_qc"]),
    "plot_mph_metrics": set(["generate_mph"]),
    "image_stitching": set(["extract_tiffs"]),
}


# If FovCallbacks ever should pass data to RunCallbacks, make this a dataclass following the
# field structure outlined for __fov_data and __panel in FovCallbacks
@dataclass
class RunCallbacks:
    """Class for run level callbacks in watcher."""

    run_folder: str

    def plot_qc_metrics(self, qc_out_dir: str, warn_overwrite=False, **kwargs):
        """Plots qc metrics generated by the `generate_qc` callback.

        Args:
            qc_out_dir (str):
                Directory containing qc metric csv
            warn_overwrite (bool): whether to warn if existing `_combined.csv` file found,
                needed to curb watcher output if `plot_qc_metrics` set as intermediate callback
            **kwargs (Dict[str, Any]):
                Additional arguments for `toffy.qc_comp.visualize_qc_metrics`.
                Accepted kwargs are

             - axes_size
             - wrap
             - dpi
             - save_dir
        Returns:
            dict:
                Maps each metric name to their respective plot
        """
        # filter kwargs
        valid_kwargs = ["axes_size", "wrap", "dpi", "save_dir"]
        viz_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}
        qc_plots = {}

        combine_qc_metrics(qc_out_dir, warn_overwrite=warn_overwrite)
        for metric_name in QC_COLUMNS:
            qc_plots[metric_name] = visualize_qc_metrics(
                metric_name, qc_out_dir, **viz_kwargs, return_plot=True
            )

        return qc_plots

    def plot_mph_metrics(self, mph_out_dir, plot_dir, warn_overwrite=False, **kwargs):
        """Plots mph metrics generated by the `generate_mph` callback.

        Args:
            mph_out_dir (str): directory containing mph metric csv
            plot_dir (str): director to store the plot to
            warn_overwrite (bool): whether to warn if existing `_combined.csv` file found,
                needed to curb watcher output if `plot_mph_metrics` set as intermediate callback
            **kwargs (Dict[str, Any]):
                Additional arguments for `toffy.mph_comp.visualize_mph`.
                Accepted kwargs are

             - regression
        Returns:
            matplotlib.figure.Figure:
                The figure containing the MPH plot
        """
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # filter kwargs
        valid_kwargs = [
            "regression",
        ]
        viz_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}

        # set verbose to false to prevent overwrite error from popping up each FOV
        mph_df = combine_mph_metrics(mph_out_dir, return_data=True, warn_overwrite=warn_overwrite)
        mph_fig = visualize_mph(mph_df, plot_dir, **viz_kwargs, return_plot=True)

        return mph_fig

    def image_stitching(self, tiff_out_dir, **kwargs):
        """Stitches individual FOV channel images together into one tiff.

        Args:
            tiff_out_dir (str): directory containing extracted images
            **kwargs (Dict[str, Any]):
                Additional arguments for `toffy.image_stitching.stitch_images`.
                Accepted kwargs are

             - channels
        """
        # filter kwargs
        valid_kwargs = ["channels"]
        viz_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}

        stitch_images(tiff_out_dir, self.run_folder, **viz_kwargs)

    def check_incomplete_fovs(self, tiff_out_dir, **kwargs):
        """Checks for partial images (even when fully extracted).

        Args:
            tiff_out_dir (str): directory containing extracted images
            **kwargs (Dict[str, Any]):
                Additional arguments for `toffy.bin_extractions.incomplete_fov_check`.
                Accepted kwargs are

             - num_rows
             - num_channels
             - signal_percent
        Raises:
            Warning if any  FOVs have partially generated images
        """
        incomplete_fov_check(self.run_folder, tiff_out_dir)

    def check_missing_fovs(self, **kwargs):
        """Checks for associated bin/json files per FOV.

        Raises:
            Warning if any fov data is missing
            **kwargs (Dict[str, Any]):
                Additional arguments for `toffy.json_utils.missing_fov_check`.
        """
        missing_fov_check(self.run_folder, os.path.basename(self.run_folder))


@dataclass
class FovCallbacks:
    """Class for FOV level callbacks in watcher."""

    run_folder: str
    point_name: str
    overwrite: bool
    __panel: pd.DataFrame = field(default=None, init=False)
    __panel_prof: pd.DataFrame = field(default=None, init=False)
    __fov_data: xr.DataArray = field(default=None, init=False)
    __fov_data_prof: xr.DataArray = field(default=None, init=False)

    def _generate_fov_data(
        self,
        panel: pd.DataFrame,
        extract_prof: bool,
        intensities=["Au", "chan_39"],
        replace=True,
        time_res=0.0005,
        **kwargs,
    ):
        """Extracts data from bin files using the given panel.

        The data and the panel are then cached members of the FovCallbacks object

        Both the deficient and proficient extracted data and panel are computed and cached

        Args:
            panel (pd.DataFrame):
                Panel used for extraction
            extract_prof (bool):
                If set, extract proficient data
            intensities (bool | List[str]):
                Intensities argument for `mibi_bin_tools.bin_files.extract_bin_files`
            replace (bool):
                Whether to replace pulse images with intensity
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
            replace=replace,
            time_res=time_res,
        )
        self.__panel = panel

        if extract_prof:
            # adds an offset of 0.3 to 'Start' and 'Stop' columns, modifying extraction range
            # from (-0.3, 0) to (0, 0.3) for proficient extraction
            panel_prof = modify_panel_ranges(panel, start_offset=0.3, stop_offset=0.3)
            self.__fov_data_prof = extract_bin_files(
                data_dir=self.run_folder,
                out_dir=None,
                include_fovs=[self.point_name],
                panel=panel_prof,
                intensities=intensities,
                replace=replace,
                time_res=time_res,
            )
            self.__panel_prof = panel_prof

    def extract_tiffs(
        self, tiff_out_dir: str, panel: pd.DataFrame, extract_prof: bool = True, **kwargs
    ):
        """Extract tiffs into provided directory, using given panel.

        Done for both the extracted deficient and proficient data

        Args:
            tiff_out_dir (str):
                Path where tiffs are written
            panel (pd.DataFrame):
                Target mass integration ranges
            extract_prof (bool):
                If set, extract mass proficient data
            **kwargs (dict):
                Additional arguments for `mibi_bin_tools.bin_files.extract_bin_files`.
                Accepted kwargs are

             - intensities
             - replace
             - time_res
        """
        if not os.path.exists(tiff_out_dir):
            os.makedirs(tiff_out_dir)

        extracted_img_dir = os.path.join(tiff_out_dir, self.point_name)
        unextracted_chan_tiffs = []

        # in the case all images have been extracted, simply return
        if os.path.exists(extracted_img_dir) and not self.overwrite:
            all_chan_tiffs = [f"{ct}.tiff" for ct in panel["Target"]]
            extracted_chan_tiffs = io_utils.list_files(extracted_img_dir, substrs=".tiff")
            unextracted_chan_tiffs = set(all_chan_tiffs).difference(extracted_chan_tiffs)

            if len(unextracted_chan_tiffs) == 0:
                warnings.warn(f"Images already extracted for FOV {self.point_name}")
                return

        # ensure we don't re-extract channels that have already been extracted
        if unextracted_chan_tiffs and not self.overwrite:
            unextracted_chans = io_utils.remove_file_extensions(unextracted_chan_tiffs)
            panel = panel[panel["Target"].isin(unextracted_chans)]

        if self.__fov_data is None or self.__fov_data_prof is None:
            self._generate_fov_data(panel, extract_prof, **kwargs)

        intensities = kwargs.get("intensities", ["Au", "chan_39"])
        if any_true(intensities) and type(intensities) is not list:
            intensities = list(self.__fov_data.channel.values)

        _write_out(
            img_data=self.__fov_data[0, :, :, :, :].values,
            out_dir=tiff_out_dir,
            fov_name=self.point_name,
            targets=list(self.__fov_data.channel.values),
            intensities=intensities,
        )

        if extract_prof:
            _write_out(
                img_data=self.__fov_data_prof[0, :, :, :, :].values,
                out_dir=tiff_out_dir + "_proficient",
                fov_name=self.point_name,
                targets=list(self.__fov_data.channel.values),
                intensities=intensities,
            )

    def generate_qc(
        self, qc_out_dir: str, panel: pd.DataFrame = None, extract_prof: bool = True, **kwargs
    ):
        """Generates qc metrics from given panel, and saves output to provided directory.

        Args:
            qc_out_dir (str):
                Path where qc_metrics are written
            panel (pd.DataFrame):
                Target mass integration ranges
            extract_prof (bool):
                If set, extract mass proficient data
            **kwargs (dict):
                Additional arguments for `toffy.qc_comp.compute_qc_metrics`. Accepted kwargs are:

             - gaussian_blur
             - blur_factor
        """
        if not os.path.exists(qc_out_dir):
            os.makedirs(qc_out_dir)

        if self.__fov_data is None:
            if panel is None:
                raise ValueError("Must provide panel if fov data is not already generated...")
            self._generate_fov_data(panel, extract_prof, **kwargs)

        qc_metric_paths = [
            os.path.join(qc_out_dir, f"{self.point_name}_nonzero_mean_stats.csv"),
            os.path.join(qc_out_dir, f"{self.point_name}_total_intensity_stats.csv"),
            os.path.join(qc_out_dir, f"{self.point_name}_percentile_99_9_stats.csv"),
        ]
        if all([os.path.exists(qc_file) for qc_file in qc_metric_paths]) and not self.overwrite:
            warnings.warn(f"All QC metrics already extracted for FOV {self.point_name}")
            return

        metric_data = compute_qc_metrics_direct(
            image_data=self.__fov_data,
            fov_name=self.point_name,
            gaussian_blur=kwargs.get("gaussian_blur", False),
            blur_factor=kwargs.get("blur_factor", 1),
        )

        for metric_name, data in metric_data.items():
            data.to_csv(os.path.join(qc_out_dir, metric_name), index=False)

    def generate_mph(self, mph_out_dir, **kwargs):
        """Generates mph metrics from given panel, and saves output to provided directory.

        Args:
            mph_out_dir (str): where to output mph csvs to
            **kwargs (dict):
                Additional arguments for `toffy.mph_comp.compute_mph_metrics`. Accepted kwargs are:

             - mass
             - mass_start
             - mass_stop
        """
        if not os.path.exists(mph_out_dir):
            os.makedirs(mph_out_dir)

        mph_pulse_file = os.path.join(mph_out_dir, f"{self.point_name}-mph_pulse.csv")
        if os.path.exists(mph_pulse_file) and not self.overwrite:
            warnings.warn(f"MPH pulse metrics already extracted for FOV {self.point_name}")
            return

        compute_mph_metrics(
            bin_file_dir=self.run_folder,
            csv_dir=mph_out_dir,
            fov=self.point_name,
            mass=kwargs.get("mass", 98),
            mass_start=kwargs.get("mass_start", 97.5),
            mass_stop=kwargs.get("mass_stop", 98.5),
        )

    def generate_pulse_heights(self, pulse_out_dir: str, panel: pd.DataFrame = None, **kwargs):
        """Generates pulse height csvs from bin files, and saves output to provided directory.

        Args:
            pulse_out_dir (str): where to output pulse height csvs
            panel (pd.DataFrame): Target mass integration ranges
            **kwargs (dict):
                Additional arguments for `toffy.normalize.write_mph_per_mass`. Accepted kwargs are:

             - start_offset
             - stop_offset
        """
        if not os.path.exists(pulse_out_dir):
            os.makedirs(pulse_out_dir)

        pulse_height_file = os.path.join(pulse_out_dir, f"{self.point_name}_pulse_heights.csv")
        if os.path.exists(pulse_height_file) and not self.overwrite:
            warnings.warn(f"Pulse heights per mass already extracted for FOV {self.point_name}")
            return

        write_mph_per_mass(
            base_dir=self.run_folder,
            output_dir=pulse_out_dir,
            fov=self.point_name,
            masses=panel["Mass"].values,
            start_offset=kwargs.get("mass_start", 0.3),
            stop_offset=kwargs.get("mass_stop", 0),
        )


def build_fov_callback(*args, **kwargs):
    """Assembles callbacks to be run for each transferred FoV.

    Args:
        *args (List[str]):
            Names of member functions of `FovCallbacks` to chain together
        **kwargs (Dict[str, Any]):
            Arguments to pass to `FovCallbacks` member functions specified in *args

    Raises:
        ValueError:
            Raised on non-existant member function or missing required kwarg

    Returns:
        Callable[[str, str], None]
            Chained fov callback which will execute all specified callbacks
    """
    # retrieve all 'non-special' methods of FovCallbacks
    methods = [attr for attr in dir(FovCallbacks) if attr[0] != "_"]

    # validate user callback settings
    misc_utils.verify_in_list(arg_strings=args, valid_callbacks=methods)
    for arg in args:
        # check that required (non-keyword) arguments for `arg` is present in passed `**kwargs`
        argnames = inspect.getfullargspec(getattr(FovCallbacks, arg))[0]
        argnames = [argname for argname in argnames if argname != "self"]
        misc_utils.verify_in_list(required_arguments=argnames, passed_arguments=list(kwargs.keys()))

    # construct actual callback
    def fov_callback(run_folder: str, point_name: str, overwrite: bool = False):
        # construct FovCallback object for given FoV
        callback_obj = FovCallbacks(run_folder, point_name, overwrite)

        # for each member, retrieve the member function and run it
        for arg in args:
            if cb := getattr(callback_obj, arg, None):
                cb(**kwargs)
            else:
                # unreachable...
                raise ValueError(f"Could not locate attribute {arg} in FovCallback object")

    return fov_callback


def build_callbacks(
    run_callbacks: Iterable[str],
    intermediate_callbacks: Iterable[str] = None,
    fov_callbacks: Iterable[str] = ("extract_tiffs",),
    **kwargs,
):
    """Deduces and assembles all run & FoV callbacks for the watcher function.

    Args:
        run_callbacks (Iterable[str]):
            List of run callback names.  These will deduce the prerequisite fov callbacks
        intermediate_callbacks (Iterable[str]):
            List of intermediate callback names, these will be subsets of `run_callbacks`
            but overriden to act as `fov_callbacks`
        fov_callbacks (Iterable[str]):
            List of fov callbacks to be run, regardless of prerequisite status
        **kwargs (Dict[str, Any]):
            Arguments to pass to `RunCallbacks` and `FovCallbacks` member functions

    Raises:
        ValueError:
            Raised on non-existant member function or missing required kwarg

    Returns:
        Callable[[None,], None], Callable[[str, str], None]:
            Assembled run callback and fov callback
    """
    methods = [attr for attr in dir(RunCallbacks) if attr[0] != "_"]

    fov_callbacks = set(fov_callbacks)

    misc_utils.verify_in_list(requested_callbacks=run_callbacks, valid_callbacks=methods)
    if intermediate_callbacks:
        misc_utils.verify_in_list(
            intermediate_callbacks=intermediate_callbacks, valid_callbacks=methods
        )

    callbacks_with_prereq = (
        run_callbacks + intermediate_callbacks if intermediate_callbacks else run_callbacks[:]
    )

    for run_cb in callbacks_with_prereq:
        argnames = inspect.getfullargspec(getattr(RunCallbacks, run_cb))[0]
        argnames = [argname for argname in argnames if argname != "self"]

        misc_utils.verify_in_list(required_arguments=argnames, passed_arguments=list(kwargs.keys()))

        fov_callbacks = fov_callbacks.union(RUN_PREREQUISITES.get(run_cb, set()))

    fov_callback = build_fov_callback(*list(fov_callbacks), **kwargs)

    def run_callback(run_folder: str):
        callback_obj = RunCallbacks(run_folder)

        for run_cb in run_callbacks:
            if cb := getattr(callback_obj, run_cb, None):
                cb(**kwargs)
            else:
                # unreachable...
                raise ValueError(f"Could not locate attribute {run_cb} in RunCallbacks object")

    intermediate_callback = None
    if intermediate_callbacks:

        def intermediate_callback(run_folder: str):
            callback_obj = RunCallbacks(run_folder)
            inter_return_vals = {}

            for run_cb in intermediate_callbacks:
                if cb := getattr(callback_obj, run_cb, None):
                    inter_return_vals[cb.__func__.__name__] = cb(**kwargs)
                else:
                    # unreachable...
                    raise ValueError(f"Could not locate attribute {run_cb} in RunCallbacks object")

            return inter_return_vals

    return fov_callback, run_callback, intermediate_callback
