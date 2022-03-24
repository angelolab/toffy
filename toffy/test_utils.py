from typing import List
import os
from pathlib import Path

import pytest
from pytest_cases import case

import pandas as pd

from toffy.settings import QC_SUFFIXES


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
DEFAULT_TAGS = ('extract', 'qc')


class ExtractionQCGenerationCases:
    @pytest.mark.xfail(raises=TypeError)
    @case(tags=DEFAULT_TAGS)
    def case_bad_global(self):
        return (-0.3, 0.0), {}

    @case(tags=DEFAULT_TAGS)
    def case_default(self):
        _, kwargs = self.case_bad_global()
        panel_path = os.path.join(Path(__file__).parent, 'data', 'sample_panel_tissue.csv')
        return pd.read_csv(panel_path), kwargs

    @case(tags='extract')
    def case_extraction_intensities(self):
        panel, kwargs = self.case_default()
        kwargs['intensities'] = True
        return panel, kwargs

    @pytest.mark.xfail()
    @case(tags=DEFAULT_TAGS)
    def case_bad_kwarg(self):
        panel, kwargs = self.case_default()
        kwargs['fake kwarg'] = "i shouldn't exist :("
        return panel, kwargs


def check_extraction_dir_structure(ext_dir: str, point_names: List[str], channels: List[str],
                                   intensities: bool = False):
    """checks extraction directory for minimum expected structure
    """
    for point in point_names:
        for channel in channels:
            assert(os.path.exists(os.path.join(ext_dir, point, f'{channel}.tiff')))

        if intensities:
            assert(os.path.exists(os.path.join(ext_dir, point, 'intensities')))


def check_qc_dir_structure(out_dir: str, point_names: List[str]):
    for point in point_names:
        for ms in QC_SUFFIXES:
            assert(os.path.exists(os.path.join(out_dir, f'{point}_{ms}.csv')))


# calling cases for the built extraction/qc callback
# this should be limited to folders to call; no generation parameters allowed  >:(
class ExtractionQCCallCases:
    def case_tissue(self):
        return os.path.join(Path(__file__).parent, 'data', 'tissue')

    def case_moly(self):
        return os.path.join(Path(__file__).parent, 'data', 'moly')
