from typing import List
import os
import shutil
import functools
from pathlib import Path
import tempfile
import json

import pytest
from pytest_cases import parametrize

import pandas as pd

from toffy.settings import QC_COLUMNS, QC_SUFFIXES
from toffy.fov_watcher import RunStructure


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
        "centerPointMicrons": {
            "x": coord[0],
            "y": coord[1]
        },
        "timingChoice": 7,
        "frameSizePixels": {
            "width": 2048,
            "height": 2048
        },
        "imagingPreset": {
            "preset": "Normal",
            "aperture": "2",
            "displayName": "Fine",
            "defaults": {
              "timingChoice": 7
            }
        },
        "sectionId": 8201,
        "slideId": 5931,
        "name": name,
        "timingDescription": "1 ms"
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
        "fovs": []
    }

    for coord, name, size in zip(fov_coords, fov_names, fov_sizes):
        sample_fovs_list["fovs"].append(
            generate_sample_fov_tiling_entry(coord, name, size)
        )

    return sample_fovs_list


# generation parameters for the extraction/qc callback build
# this should be limited to the panel, foldernames, and kwargs
FOV_CALLBACKS = ('extract_tiffs', 'generate_qc', 'generate_mph')
RUN_CALLBACKS = ('plot_qc_metrics', 'plot_mph_metrics')


class ExtractionQCGenerationCases:
    def case_all_callbacks(self):
        panel_path = os.path.join(Path(__file__).parent, 'data', 'sample_panel_tissue.csv')
        return FOV_CALLBACKS, {'panel': pd.read_csv(panel_path)}

    def case_extract_only(self):
        cbs, kwargs = self.case_all_callbacks()
        return cbs[:1], kwargs

    def case_qc_only(self):
        cbs, kwargs = self.case_all_callbacks()
        return cbs[1:2], kwargs

    def case_mph_only(self):
        cbs, kwargs = self.case_all_callbacks()
        return cbs[2:3], kwargs

    def case_extraction_intensities(self):
        cbs, kwargs = self.case_all_callbacks()
        kwargs['intensities'] = True
        return cbs, kwargs

    @pytest.mark.xfail(raises=ValueError)
    def case_missing_panel(self):
        cbs, _ = self.case_all_callbacks()
        return cbs, {}

    @pytest.mark.xfail(raises=ValueError)
    def case_bad_callback(self):
        return ['invalid_callback'], {}


class PlotQCMetricsCases:
    def case_default(self):
        panel_path = os.path.join(Path(__file__).parent, 'data', 'sample_panel_tissue.csv')
        return RUN_CALLBACKS, {'panel': pd.read_csv(panel_path)}

    def save_figure(self):
        cbs, kws = self.case_default()
        kws['save_dir'] = True
        return cbs, kws

    @pytest.mark.xfail(raises=ValueError)
    def case_missing_panel(self):
        cbs, _ = self.case_default()
        return cbs, {}

    @pytest.mark.xfail(raises=ValueError)
    def case_bad_callback(self):
        return ['invalid_callback'], {}


def check_extraction_dir_structure(ext_dir: str, point_names: List[str], channels: List[str],
                                   intensities: bool = False):
    """Checks extraction directory for minimum expected structure

    Args:
        ext_dir (str):
            Folder containing extraction output
        point_names (list):
            List of expected point names
        channels (list):
            List of expected channel names
        intensities (bool):
            Whether or not to check for intensities

    Raises:
        AssertionError:
            Assertion error on missing expected tiff
    """
    for point in point_names:
        for channel in channels:
            assert(os.path.exists(os.path.join(ext_dir, point, f'{channel}.tiff')))

        if intensities:
            assert(os.path.exists(os.path.join(ext_dir, point, 'intensities')))


def check_qc_dir_structure(out_dir: str, point_names: List[str], qc_plots: bool = False):
    """Checks QC directory for minimum expected structure

    Args:
        out_dir (str):
            Folder containing QC output
        point_names (list):
            List of expected point names
        qc_plots (bool):
            Whether to expect plot files

    Raises:
        AssertionError:
            Assertion error on missing csv
    """
    for point in point_names:
        for mn, ms in zip(QC_COLUMNS, QC_SUFFIXES):
            assert(os.path.exists(os.path.join(out_dir, f'{point}_{ms}.csv')))
            if qc_plots:
                assert(os.path.exists(os.path.join(out_dir, '%s_barplot_stats.png' % mn)))


def check_mph_dir_structure(img_out_dir: str, point_names: List[str], combined: bool = False):
    """Checks MPH directory for minimum expected structure

    Args:
        img_out_dir (str):
            Folder containing MPH plot output
        point_names (list):
            List of expected point names
        combined (bool):
            whether to check for combined mph data csv and plot image

    Raises:
        AssertionError:
            Assertion error on missing csv
    """
    for point in point_names:
        assert(os.path.exists(os.path.join(img_out_dir, f'{point}-pulse_height.csv')))

    if combined:
        assert(os.path.exists(os.path.join(img_out_dir, 'total_count_vs_mph_data.csv')))
        assert(os.path.exists(os.path.join(img_out_dir, 'fov_vs_mph.jpg')))


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
        ex_fov = {
            "scanCount": scan_count,
            "runOrder": run_order,
            "name": name
        }
        fov_list.append(ex_fov)

    # delete name key if one is not provided
    for fov in sample_run.get('fovs', ()):
        if fov.get('name') is None:
            del fov['name']

    # create bad dictionary
    if bad:
        sample_run['bad key'] = sample_run['fovs']
        del sample_run['fovs']

    # create json file for the data
    if create_json:
        temp = tempfile.NamedTemporaryFile(mode="w", delete=False)
        json.dump(sample_run, temp)
        return temp.name

    return sample_run


# calling cases for the built extraction/qc callback
# this should be limited to folders to call; no generation parameters allowed  >:(
class ExtractionQCCallCases:
    def case_tissue(self):
        return os.path.join(Path(__file__).parent, 'data', 'tissue')

    def case_moly(self):
        return os.path.join(Path(__file__).parent, 'data', 'moly')


def _make_small_file(folder: str, name: str):
    """Creates small file.  Creating a blank file will cause a stall for 0-size checking

    Args:
        folder (str):
            Folder to store file in
        name (str):
            Name of small file
    """
    with open(os.path.join(folder, name), 'w') as f:
        f.write('a')


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
        with open(os.path.join(self.tmpdir, f'{self.tempdir_name}.json'), 'w') as f:
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
        run_json = {
            'fovs': [
                {'runOrder': n + 1, 'scanCount': 1}
                for n in range(fov_count)
            ]
        }
        expected_files = [
            f'fov-{n + 1}-scan-1.bin'
            for n in range(fov_count)
        ]
        expected_files += [binf.split('.')[0] + '.json' for binf in expected_files]
        return run_json, expected_files

    @pytest.mark.xfail(raises=KeyError)
    def case_extrabin(self):
        run_json, exp_files = self.case_default()
        exp_files += ['fov-X-scan-1.bin']
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
    def case_default(self, intensity):
        panel = pd.read_csv(os.path.join(Path(__file__).parent, 'data', 'sample_panel_tissue.csv'))
        validators = [
            functools.partial(
                check_extraction_dir_structure,
                channels=list(panel['Target']),
                intensities=intensity),
            check_qc_dir_structure,
            check_mph_dir_structure,
        ]

        kwargs = {'panel': panel, 'intensities': intensity}

        return (
            ['plot_qc_metrics'],
            ['extract_tiffs'],
            ['plot_mph_metrics'],
            kwargs,
            validators
        )
