import functools
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from alpineer.test_utils import _make_small_file
from pytest_cases import parametrize

from toffy.fov_watcher import RunStructure
from toffy.json_utils import write_json_file
from toffy.settings import QC_COLUMNS, QC_SUFFIXES


def make_run_file(tmp_dir, prefixes=[], include_nontiled=False):
    """Create a run subir and run json in the provided dir and return the path to this new dir."""

    if len(prefixes) == 1:
        prefix1, prefix2 = prefixes * 2
    else:
        prefix1 = prefixes[0]
        prefix2 = prefixes[1]

    fov_data = {
        f"{prefix1}R1C3": 10,
        f"{prefix1}R2C1": 8,
        "MoQC": 20,
        f"{prefix2}R2C2": 8,
    }
    if include_nontiled:
        fov_data["nontiled"] = 8

    run_data = []
    for i, fov in enumerate(fov_data.keys()):
        run_data.append(
            {
                "runOrder": i + 1,
                "scanCount": 1,
                "frameSizePixels": {"width": fov_data[fov], "height": fov_data[fov]},
                "name": fov,
            }
        )
    run_json_spoof = {"fovs": run_data}

    test_dir = os.path.join(tmp_dir, "data", "test_run")
    os.makedirs(test_dir)
    json_path = os.path.join(test_dir, "test_run.json")
    write_json_file(json_path, run_json_spoof)

    return test_dir


def generate_sample_fov_tiling_entry(coord, name, size):
    """Generates a sample fov entry to put in a sample fovs list for tiling

    Args:
        coord (tuple):
            Defines the starting x and y point for the fov
        name (str):
            Defines the name of the fov
        size (int):
            Defines the size along both axes of each fov

    Returns:
        dict:
            An entry to be placed in the fovs list with provided coordinate and name
    """

    sample_fov_tiling_entry = {
        "scanCount": 1,
        "fovSizeMicrons": size,
        "centerPointMicrons": {"x": coord[0], "y": coord[1]},
        "timingChoice": 7,
        "frameSizePixels": {"width": 2048, "height": 2048},
        "imagingPreset": {
            "preset": "Normal",
            "aperture": "2",
            "displayName": "Fine",
            "defaults": {"timingChoice": 7},
        },
        "sectionId": 8201,
        "slideId": 5931,
        "name": name,
        "timingDescription": "1 ms",
    }

    return sample_fov_tiling_entry


def generate_sample_fovs_list(fov_coords, fov_names, fov_sizes):
    """Generate a sample dictionary of fovs for tiling

    Args:
        fov_coords (list):
            A list of tuples listing the starting x and y coordinates of each fov
        fov_names (list):
            A list of strings identifying the name of each fov
        fov_sizes (list):
            A list of ints identifying the size in microns of each fov along both axes

    Returns:
        dict:
            A dummy fovs list with starting x and y set to the provided coordinates and name
    """

    sample_fovs_list = {
        "exportDateTime": "2021-03-12T19:02:37.920Z",
        "fovFormatVersion": "1.5",
        "fovs": [],
    }

    for coord, name, size in zip(fov_coords, fov_names, fov_sizes):
        sample_fovs_list["fovs"].append(generate_sample_fov_tiling_entry(coord, name, size))

    return sample_fovs_list


# generation parameters for the extraction/qc callback build
# this should be limited to the panel, foldernames, and kwargs
FOV_CALLBACKS = (
    "extract_tiffs",
    "generate_qc",
    "generate_mph",
    "generate_pulse_heights",
)
RUN_CALLBACKS = ("plot_qc_metrics", "plot_mph_metrics", "image_stitching")


def mock_visualize_qc_metrics(
    metric_name,
    qc_metric_dir,
    return_plot=True,
    axes_size=16,
    wrap=6,
    dpi=None,
    save_dir=None,
    ax=None,
):
    if save_dir:
        _make_small_file(save_dir, "%s_barplot_stats.png" % metric_name)

    if return_plot:
        qc_metric_df = pd.DataFrame(np.zeros((5, 3)), columns=["fov", metric_name, "channel"])
        qc_metric_df["fov"] = "fov-1-scan-1"
        qc_metric_df["channel"] = [f"chan{i}" for i in np.arange(5)]
        qc_fg = sns.catplot(
            x="fov",
            y=metric_name,
            col="channel",
            col_wrap=1,
            data=qc_metric_df,
            kind="bar",
            color="black",
            sharex=True,
            sharey=False,
        )
        return qc_fg


def mock_visualize_mph(mph_df, out_dir, return_plot=True, regression: bool = False):
    if out_dir:
        _make_small_file(out_dir, "fov_vs_mph.jpg")

    if return_plot:
        mph_df = pd.DataFrame(np.random.rand(5, 2), columns=["cum_total_count", "MPH"])
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(mph_df["cum_total_count"], mph_df["MPH"])
        ax2 = ax1.twiny()
        return fig


class FovCallbackCases:
    def case_all_callbacks(self):
        panel_path = os.path.join(Path(__file__).parents[2], "data", "sample_panel.csv")
        return FOV_CALLBACKS, {"panel": pd.read_csv(panel_path), "extract_prof": True}

    def case_dont_extract_prof(self):
        cbs, kwargs = self.case_all_callbacks()
        kwargs["extract_prof"] = False
        return cbs, kwargs

    def case_extract_only(self):
        cbs, kwargs = self.case_all_callbacks()
        return cbs[:1], kwargs

    def case_qc_only(self):
        cbs, kwargs = self.case_all_callbacks()
        return cbs[1:2], kwargs

    def case_mph_only(self):
        cbs, kwargs = self.case_all_callbacks()
        return cbs[2:3], kwargs

    def case_pulse_heights_only(self):
        cbs, kwargs = self.case_all_callbacks()
        return cbs[3:4], kwargs

    def case_extraction_intensities(self):
        cbs, kwargs = self.case_all_callbacks()
        kwargs["intensities"] = True
        kwargs["replace"] = True
        return cbs, kwargs

    def case_extraction_intensities_not_replace(self):
        cbs, kwargs = self.case_all_callbacks()
        kwargs["intensities"] = True
        kwargs["replace"] = False
        return cbs, kwargs

    @pytest.mark.xfail(raises=ValueError)
    def case_missing_panel(self):
        cbs, _ = self.case_all_callbacks()
        return cbs, {}

    @pytest.mark.xfail(raises=ValueError)
    def case_bad_callback(self):
        return ["invalid_callback"], {}


class RunCallbackCases:
    def case_default(self):
        panel_path = os.path.join(Path(__file__).parents[2], "data", "sample_panel.csv")
        return RUN_CALLBACKS, None, {"panel": pd.read_csv(panel_path), "extract_prof": True}

    def save_figure(self):
        cbs, ibs, kws = self.case_default()
        kws["save_dir"] = True
        return cbs, ibs, kws

    def case_dont_extract_prof(self):
        cbs, ibs, kws = self.case_default()
        kws["extract_prof"] = False
        return cbs, ibs, kws

    def case_inter_callback(self):
        cbs, ibs, kws = self.case_default()
        ibs = list(cbs[:2])
        cbs = list(cbs[2:])
        return cbs, ibs, kws

    @pytest.mark.xfail(raises=ValueError)
    def case_missing_panel(self):
        cbs, _, _ = self.case_default()
        return cbs, None, {}

    @pytest.mark.xfail(raises=ValueError)
    def case_bad_run_callback(self):
        return ["invalid_callback"], None, {}

    @pytest.mark.xfail(raises=ValueError)
    def case_bad_inter_callback(self):
        return RUN_CALLBACKS, ["invalid_callback"], {}


def check_extraction_dir_structure(
    ext_dir: str,
    point_names: List[str],
    bad_points: List[str],
    channels: List[str],
    intensities: bool = False,
    replace: bool = True,
):
    """Checks extraction directory for minimum expected structure

    Args:
        ext_dir (str):
            Folder containing extraction output
        point_names (list):
            List of expected point names
        bad_points (list):
            list of points which should not have structure created
        channels (list):
            List of expected channel names
        intensities (bool):
            Whether or not to check for intensities
        replace (bool):
            Whether to replace pulse images with intensity

    Raises:
        AssertionError:
            Assertion error on missing expected tiff
    """

    for point, bad in zip(point_names, bad_points):
        assert not os.path.exists(os.path.join(ext_dir, bad))

        for channel in channels:
            assert os.path.exists(os.path.join(ext_dir, point, f"{channel}.tiff"))

        if intensities and not replace:
            assert os.path.exists(os.path.join(ext_dir, point, "intensities"))


def check_qc_dir_structure(
    out_dir: str, point_names: List[str], bad_points: List[str], qc_plots: bool = False
):
    """Checks QC directory for minimum expected structure

    Args:
        out_dir (str):
            Folder containing QC output
        point_names (list):
            List of expected point names
        bad_points (list):
            list of points which should not have structure created
        qc_plots (bool):
            Whether to expect plot files

    Raises:
        AssertionError:
            Assertion error on missing csv
    """
    for point, bad in zip(point_names, bad_points):
        for mn, ms in zip(QC_COLUMNS, QC_SUFFIXES):
            assert os.path.exists(os.path.join(out_dir, f"{point}_{ms}.csv"))
            assert not os.path.exists(os.path.join(out_dir, f"{bad}_{ms}.csv"))
            if qc_plots:
                assert os.path.exists(os.path.join(out_dir, "%s_barplot_stats.png" % mn))


def check_mph_dir_structure(
    mph_out_dir: str,
    plot_dir: str,
    point_names: List[str],
    bad_points: List[str],
    combined: bool = False,
):
    """Checks MPH directory for minimum expected structure

    Args:
        mph_out_dir (str):
            Folder containing the MPH csv files
        plot_dir (str):
            Folder containing MPH plot output
        point_names (list):
            List of expected point names
        bad_points (list):
            list of points which should not have structure created
        combined (bool):
            whether to check for combined mph data csv and plot image

    Raises:
        AssertionError:
            Assertion error on missing csv
    """
    for point, bad in zip(point_names, bad_points):
        assert os.path.exists(os.path.join(mph_out_dir, f"{point}-mph_pulse.csv"))
        assert not os.path.exists(os.path.join(mph_out_dir, f"{bad}-mph_pulse.csv"))

    if combined:
        assert os.path.exists(os.path.join(mph_out_dir, "mph_pulse_combined.csv"))
        assert os.path.exists(os.path.join(plot_dir, "fov_vs_mph.png"))


def check_pulse_dir_structure(pulse_out_dir: str, point_names: List[str], bad_points: List[str]):
    """Checks pulse heights directory for minimum expected structure

    Done for both proficient and deficient pulse height data

    Args:
        pulse_out_dir (str):
            Folder containing pulse height files
        point_names (list):
            List of expected point names
        bad_points (list):
            list of points which should not have structure created

    Raises:
        AssertionError:
            Assertion error on missing csv
    """

    for point, bad in zip(point_names, bad_points):
        assert os.path.exists(os.path.join(pulse_out_dir, f"{point}_pulse_heights.csv"))
        assert not os.path.exists(os.path.join(pulse_out_dir, f"{bad}_pulse_heights.csv"))


def check_stitched_dir_structure(stitched_dir: str, channels: List[str]):
    """Checks extraction directory for stitching structure

    Args:
        stitched_dir (str):
            Folder containing stitched output
        channels (list):
            List of expected channel names

    Raises:
        AssertionError:
            Assertion error on missing expected tiff
    """
    for channel in channels:
        assert os.path.exists(os.path.join(stitched_dir, f"{channel}_stitched.tiff"))


def create_sample_run(name_list, run_order_list, scan_count_list, create_json=False, bad=False):
    """Creates sample run metadata with option to create a temporary json file

    Args:
        name_list (list): List of strings for FOV names
        run_order_list (list): List detailing run order
        scan_count_list (list): List of the scanCount for each FOV
        create_json (bool): Whether or not to return a json tempfile path
        bad (bool): Whether or not to create a dictionary without correct FOV key setup

    Returns:
        sample_run (dict): the dictionary for the sample run metadata
        temp.name (str): path to the temporary json run file

    """
    fov_list = []
    sample_run = {"fovs": fov_list}

    # set up dictionary
    for name, run_order, scan_count in zip(name_list, run_order_list, scan_count_list):
        ex_fov = {"scanCount": scan_count, "runOrder": run_order, "name": name}
        fov_list.append(ex_fov)

    # delete name key if one is not provided
    for fov in sample_run.get("fovs", ()):
        if fov.get("name") is None:
            del fov["name"]

    # create bad dictionary
    if bad:
        sample_run["bad key"] = sample_run["fovs"]
        del sample_run["fovs"]

    # create json file for the data
    if create_json:
        temp = tempfile.NamedTemporaryFile(mode="w", delete=False)
        json.dump(sample_run, temp)
        return temp.name

    return sample_run


# calling cases for data to test against
# this should be limited to folders to call; no generation parameters allowed  >:(
class WatcherTestData:
    def case_combined(self):
        return os.path.join(Path(__file__).parents[2], "data", "combined")


class RunStructureTestContext:
    def __init__(self, run_json, files=None):
        self.run_json = run_json
        self.files = files
        self.tmpdir = None
        self.run_structure = None

    @property
    def tempdir_name(self):
        return Path(self.tmpdir).parts[-1]

    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp()
        with open(os.path.join(self.tmpdir, f"{self.tempdir_name}.json"), "w") as f:
            json.dump(self.run_json, f)

        for file in self.files:
            _make_small_file(self.tmpdir, file)

        # build run structure dir
        self.run_structure = RunStructure(self.tmpdir)
        return self.tmpdir, self.run_structure

    def __exit__(self, exc_type, exc_value, exc_traceback):
        shutil.rmtree(self.tmpdir)


class RunStructureCases:
    def case_default(self):
        fov_count = 2
        run_json = {"fovs": [{"runOrder": n + 1, "scanCount": 1} for n in range(fov_count)]}
        expected_files = [f"fov-{n + 1}-scan-1.bin" for n in range(fov_count)]
        expected_files += [binf.split(".")[0] + ".json" for binf in expected_files]
        return run_json, expected_files

    @pytest.mark.xfail(raises=KeyError)
    def case_extrabin(self):
        run_json, exp_files = self.case_default()
        exp_files += ["fov-X-scan-1.bin"]
        return run_json, exp_files


class WatcherCases:
    """Test cases for start_watcher

    Cases in this class will, in order, return:
        (fov callback names, run callback names, kwargs for the callbacks,
         directory validation functions)

    Required directory kwargs must be added within the actual test function. They aren't added
    here since contructing the temp directory within this class would probably be harder to manage.

    Validation functions will check that the directory/files created by each callback are correct.
    Some maybe partialed for convinience, since some arguments, like the panel, are created here
    and might as well be premptively passed, so that only "run dependent" arguments need to be
    passed to the validators (i.e out directory and point names).
    """

    @parametrize(intensity=(False, True))
    @parametrize(replace=(False, True))
    @parametrize(extract_prof=(False, True))
    def case_default(self, intensity, replace, extract_prof):
        panel = pd.read_csv(os.path.join(Path(__file__).parents[2], "data", "sample_panel.csv"))
        validators = [
            functools.partial(
                check_extraction_dir_structure,
                channels=list(panel["Target"]),
                intensities=intensity,
                replace=replace,
            ),
            check_qc_dir_structure,
            check_mph_dir_structure,
            functools.partial(check_stitched_dir_structure, channels=list(panel["Target"])),
            check_pulse_dir_structure,
        ]

        kwargs = {
            "panel": panel,
            "extract_prof": extract_prof,
            "intensities": intensity,
            "replace": replace,
        }

        return (
            ["plot_qc_metrics", "plot_mph_metrics", "image_stitching"],
            None,
            ["extract_tiffs", "generate_pulse_heights"],
            kwargs,
            validators,
        )

    @parametrize(intensity=(False, True))
    @parametrize(replace=(False, True))
    @parametrize(extract_prof=(False, True))
    def case_inter_callback(self, intensity, replace, extract_prof):
        rcs, _, fcs, kwargs, validators = self.case_default(intensity, replace, extract_prof)
        ics = rcs[:2]
        rcs = rcs[2:]

        return (rcs, ics, fcs, kwargs, validators)
