import json
import os
import tempfile
from unittest.mock import call, patch

import numpy as np
import pytest
from alpineer.test_utils import _make_blank_file

from toffy import json_utils

from .utils import test_utils


def test_rename_missing_fovs():
    # data with missing names
    ex_name = ["custom_1", None, "custom_2", "custom_3", None]
    ex_run_order = list(range(1, 6))
    ex_scan_count = list(range(1, 6))

    # create a dict with the sample data
    ex_run = test_utils.create_sample_run(ex_name, ex_run_order, ex_scan_count)

    # test that missing names are given a placeholder
    ex_run = json_utils.rename_missing_fovs(ex_run)
    for fov in ex_run.get("fovs", ()):
        assert fov.get("name") is not None


def test_rename_duplicate_fovs():
    # define a sample set of FOVs
    fov_coords = [(100 + 100 * i, 100 + 100 * j) for i in np.arange(4) for j in np.arange(3)]
    fov_names = ["R%dC%d" % (i, j) for i in np.arange(1, 5) for j in np.arange(1, 4)]
    fov_sizes = [1000] * 12
    fov_list = test_utils.generate_sample_fovs_list(fov_coords, fov_names, fov_sizes)

    # no duplicate FOV names identified, no names should be changed
    fov_list_no_dup = json_utils.rename_duplicate_fovs(fov_list)
    assert [fov["name"] for fov in fov_list_no_dup["fovs"]] == fov_names

    # rename R2C2 and R2C3 as R2C1 to create one set of duplicates
    fov_list["fovs"][4]["name"] = "R2C1"
    fov_list["fovs"][5]["name"] = "R2C1"
    fov_names[4] = "R2C1_duplicate1"
    fov_names[5] = "R2C1_duplicate2"

    fov_list_one_dup = json_utils.rename_duplicate_fovs(fov_list)
    assert [fov["name"] for fov in fov_list_one_dup["fovs"]] == fov_names

    # rename R3C3, R4C1, and R4C2 as R3C2 to create another set of duplicates of differing size
    fov_list["fovs"][8]["name"] = "R3C2"
    fov_list["fovs"][9]["name"] = "R3C2"
    fov_list["fovs"][10]["name"] = "R3C2"
    fov_names[8] = "R3C2_duplicate1"
    fov_names[9] = "R3C2_duplicate2"
    fov_names[10] = "R3C2_duplicate3"

    fov_list_mult_dup = json_utils.rename_duplicate_fovs(fov_list)
    assert [fov["name"] for fov in fov_list_mult_dup["fovs"]] == fov_names


def test_list_moly_fovs(tmpdir):
    # create fake jsons
    moly_json = {"name": "bob", "standardTarget": "Molybdenum Foil"}

    tissue_json = {"name": "carl"}

    # create list of moly and non-moly FOVs
    moly_fovs = ["fov-1-scan-1", "fov-2-scan-2", "fov-2-scan-3"]
    tissue_fovs = ["fov-1-scan-2", "fov-3-scan-1"]

    # save jsons
    for fov in moly_fovs:
        json_path = os.path.join(tmpdir, fov + ".json")
        with open(json_path, "w") as jp:
            json.dump(moly_json, jp)

    for fov in tissue_fovs:
        json_path = os.path.join(tmpdir, fov + ".json")
        with open(json_path, "w") as jp:
            json.dump(tissue_json, jp)

    # run file json
    _make_blank_file(tmpdir, os.path.basename(tmpdir) + ".json")

    pred_moly_fovs = json_utils.list_moly_fovs(tmpdir)

    assert np.array_equal(pred_moly_fovs.sort(), moly_fovs.sort())

    # check fov_list functionality
    pred_moly_fovs_subset = json_utils.list_moly_fovs(tmpdir, ["fov-1-scan-1", "fov-1-scan-2"])

    assert np.array_equal(pred_moly_fovs_subset.sort(), ["fov-1-scan-1"].sort())


def test_read_json_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # create fake jsons
        moly_json = {"name": "bob", "standardTarget": "Molybdenum Foil"}
        json_path = tmp_dir + "/test.json"

        # write test json
        with open(json_path, "w") as jp:
            json.dump(moly_json, jp)

        # Test bad path
        bad_path = "/neasdf1246ljea/asdfje12ua3421ndsf/asdf.json"
        with pytest.raises(FileNotFoundError, match=r"A bad path*"):
            json_utils.read_json_file(bad_path)

        # Read json with read_json_file function assuming file path is good
        newfile_test = json_utils.read_json_file(json_path)

        # Make sure using the read_json_file leads to the same object as moly_json
        assert newfile_test == moly_json


def test_write_json_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # create fake jsons
        moly_json = {"name": "bob", "standardTarget": "Molybdenum Foil"}
        json_path = tmp_dir + "/test.json"

        # To be 100% you would want to create some massive random string instead of hardcode
        bad_path = "/mf8575b20d/bgjeidu45483hdck/asdf.json"

        # test bad path
        with pytest.raises(FileNotFoundError, match=r"A bad path*"):
            json_utils.write_json_file(json_path=bad_path, json_object=moly_json)

        # Write file after file path is validated
        json_utils.write_json_file(json_path=json_path, json_object=moly_json)

        # Read file with standard method
        with open(json_path, "r") as jp:
            newfile_test = json.load(jp)

        # Make sure the file written with write_json_file is the same as starting point
        assert newfile_test == moly_json


def test_split_run_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_file_name = "test.json"
        run_file = {"fovs": ["fov1", "fov2", "fov3", "fov4", "fov5", "fov6", "fov7", "fov8"]}
        json_utils.write_json_file(os.path.join(tmp_dir, run_file_name), run_file, "utf-8")

        # list not summing to fov amount will raise an error
        bad_split = [5, 5]
        with pytest.raises(ValueError, match=r"does not match the number of FOVs"):
            json_utils.split_run_file(tmp_dir, run_file_name, bad_split)

        # test successful json split
        good_split = [2, 4, 2]
        json_utils.split_run_file(tmp_dir, run_file_name, good_split)

        # check the new files exist
        new_data = {}
        for i in list(range(1, len(good_split) + 1)):
            new_json = os.path.join(tmp_dir, "test_part" + str(i) + ".json")
            new_data["test_part{0}".format(i)] = json_utils.read_json_file(new_json, "utf-8")

        # check for correct fov splitting
        assert new_data["test_part1"]["fovs"] == ["fov1", "fov2"]
        assert new_data["test_part2"]["fovs"] == ["fov3", "fov4", "fov5", "fov6"]
        assert new_data["test_part3"]["fovs"] == ["fov7", "fov8"]


def test_check_for_empty_files():
    test_data = [1, 2, 3, 4, 5]

    with tempfile.TemporaryDirectory() as temp_dir:
        json_utils.write_json_file(os.path.join(temp_dir, "non_empty_file.json"), test_data)
        _make_blank_file(temp_dir, "non_empty_file.bin")

        # test that no empty files detected returns empty list
        no_empty_files = json_utils.check_for_empty_files(temp_dir)
        assert no_empty_files == []

        _make_blank_file(temp_dir, "empty_file.json")
        _make_blank_file(temp_dir, "empty_file.bin")

        # test successful empty file detection:
        with pytest.warns(UserWarning, match="The following FOVs have empty json files"):
            empty_files = json_utils.check_for_empty_files(temp_dir)

        assert empty_files == ["empty_file"]


@patch("builtins.print")
def test_check_fov_resolutions(mocked_print):
    with tempfile.TemporaryDirectory() as temp_dir:
        run_data = {
            "fovs": [
                {
                    "runOrder": 1,
                    "scanCount": 1,
                    "frameSizePixels": {"width": 32, "height": 32},
                    "fovSizeMicrons": 100,
                },
                {
                    "runOrder": 2,
                    "scanCount": 1,
                    "frameSizePixels": {"width": 32, "height": 32},
                    "fovSizeMicrons": 100,
                },
                {
                    "runOrder": 3,
                    "scanCount": 1,
                    "frameSizePixels": {"width": 16, "height": 16},
                    "fovSizeMicrons": 100,
                },
                {
                    "runOrder": 4,
                    "scanCount": 1,
                    "frameSizePixels": {"width": 8, "height": 8},
                    "fovSizeMicrons": 100,
                    "standardTarget": "Molybdenum Foil",
                },
            ],
        }

        json_utils.write_json_file(os.path.join(temp_dir, "test_run.json"), run_data)

        # test successful resolution check and print statements
        resolution_data = json_utils.check_fov_resolutions(
            temp_dir, "test_run", save_path=os.path.join(temp_dir, "resolution_data.csv")
        )
        assert mocked_print.mock_calls[0] == call("Resolutions are not consistent among all FOVs.")
        assert mocked_print.mock_calls[1] == call(
            "         fov name  pixels / 400 microns\nfov-3-scan-1 None                    64"
        )

        # moly fov is ignored
        assert resolution_data.shape == (3, 3)
        assert (
            np.array(["fov-1-scan-1", "fov-2-scan-1", "fov-3-scan-1"]) == resolution_data["fov"]
        ).all()
        assert (
            resolution_data["pixels / 400 microns"].iloc[0]
            == resolution_data["pixels / 400 microns"].iloc[1]
        )
        assert (
            resolution_data["pixels / 400 microns"].iloc[0]
            != resolution_data["pixels / 400 microns"].iloc[2]
        )

        assert os.path.exists(os.path.join(temp_dir, "resolution_data.csv"))
