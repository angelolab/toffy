# future potential imports as this file gets built up
import json
import os
import sys


def generate_sample_fov_tiling_entry(coord, name):
    """Generates a sample fov entry to put in a sample fovs list for tiling

    Args:
        coord (tuple):
            Defines the starting x and y point for the fov
        name (str):
            Defines the name of the fov

    Returns:
        dict:
            An entry to be placed in the fovs list with provided coordinate and name
    """

    sample_fov_tiling_entry = {
        "scanCount": 1,
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


def generate_sample_fovs_list(fov_coords, fov_names):
    """Generate a sample dictionary of fovs for tiling

    Args:
        fov_coords (list):
            A list of tuples listing the starting x and y coordinates of each fov
        fov_names (list):
            A list of strings identifying the name of each fov

    Returns:
        dict:
            A dummy fovs list with starting x and y set to the provided coordinates and name
    """

    sample_fovs_list = {
        "exportDateTime": "2021-03-12T19:02:37.920Z",
        "fovFormatVersion": "1.5",
        "fovs": []
    }

    for coord, name in zip(fov_coords, fov_names):
        sample_fovs_list["fovs"].append(
            generate_sample_fov_tiling_entry(coord, name)
        )

    return sample_fovs_list
