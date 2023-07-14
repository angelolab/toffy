import csv
import os
import tempfile

import pandas as pd
import pytest
from alpineer import test_utils

from toffy import panel_utils


def test_modify_panel_ranges():
    # only Calprotectin's range should be adjusted
    toffy_panel = pd.DataFrame(
        {
            "Mass": [69, 71, 89],
            "Target": ["Calprotectin", "Chymase", "Mast Cell Tryptase"],
            "Start": [68.7, 70.6, 89.3],
            "Stop": [69, 71, 89],
        }
    )

    toffy_panel_pos_offset = panel_utils.modify_panel_ranges(
        toffy_panel, start_offset=0.3, stop_offset=0.3
    )

    assert list(toffy_panel_pos_offset["Start"]) == [69, 70.6, 89.3]
    assert list(toffy_panel_pos_offset["Stop"]) == [69.3, 71, 89]

    toffy_panel_neg_offset = panel_utils.modify_panel_ranges(
        toffy_panel, start_offset=-0.3, stop_offset=-0.3
    )

    assert list(toffy_panel_neg_offset["Start"]) == [68.4, 70.6, 89.3]
    assert list(toffy_panel_neg_offset["Stop"]) == [68.7, 71, 89]

    toffy_panel_one_offset = panel_utils.modify_panel_ranges(toffy_panel, stop_offset=-0.3)

    assert list(toffy_panel_one_offset["Start"]) == [68.7, 70.6, 89.3]
    assert list(toffy_panel_one_offset["Stop"]) == [68.7, 71, 89]


def test_merge_duplicate_masses():
    duplicate_panel = pd.DataFrame(
        {
            "Mass": [1, 2, 3, 3, 1, 1],
            "Target": [
                "target1",
                "target2",
                "target3",
                "target4",
                "target5",
                "target6",
            ],
        }
    )

    unique_panel = panel_utils.merge_duplicate_masses(duplicate_panel)

    # check for no duplicate masses
    assert len(list(unique_panel["Mass"])) == len(set(unique_panel["Mass"]))

    # check for correct edited target names
    assert list(unique_panel["Target"]) == [
        "target1_target5_target6",
        "target2",
        "target3_target4",
    ]


def test_convert_panel():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_panel = pd.DataFrame(
            {
                "ID (Lot)": [1352, 1352, 1350, 1351],
                "Target": [
                    "Calprotectin",
                    "Duplicate",
                    "Chymase",
                    "Mast Cell Tryptase",
                ],
                "Clone": ["MAC387", "MAC387", "EPR13136", "EPR9522"],
                "Mass": [69, 69, 71, 89],
                "Element": ["Ga", "Ga", "Ga", "Y"],
                "Manufactured": ["7/20/20", "7/20/20", "7/20/20", "5/5/21"],
                "Stock": [200, 200, 200, 200],
                "Titer": [0.125, 0.125, 0.125, 0.25],
                "Volume (μL)": [0, 0, 0, 0],
                "Staining Batch": [1, 1, 1, 1],
            }
        )
        test_panel.to_csv(os.path.join(temp_dir, "test_panel.csv"), index=False)

        # add metadata
        with open(os.path.join(temp_dir, "test_panel.csv"), newline="") as f:
            r = csv.reader(f)
            data = [line for line in r]
        with open(os.path.join(temp_dir, "test_panel.csv"), "w", newline="") as f:
            w = csv.writer(f)
            metadata = (
                '"Panel ID",106\r\n"Panel Name","DCIS_followup"'
                '\r\n"Manufacture Date","2021-08-25"\r\n"Description",""\r\n"Batch",1'
                '\r\n"Total Volume (μL)",0\r\n"Antibody Volume (μL)",0\r\n'
                '"Buffer Volume (μL)",0'
            )
            w.writerows([metadata])
            w.writerows(data)

        # test successful new panel creation
        converted_panel = panel_utils.convert_panel(os.path.join(temp_dir, "test_panel.csv"))
        assert os.path.exists(os.path.join(temp_dir, "test_panel-toffy.csv"))

        necessary_panel = panel_utils.necessary_masses

        # check toffy panel structure
        assert list(converted_panel.columns) == ["Mass", "Target", "Start", "Stop"]

        # check chan_71 is not duplicated
        assert ["chan_71", "Duplicate"] not in list(converted_panel["Target"])

        # check concatenated target names for same mass
        assert "Calprotectin_Duplicate" in list(converted_panel["Target"])

        # check for unique mass values
        assert len(list(converted_panel["Mass"])) == len(set(converted_panel["Mass"]))

        # check for all necessary masses
        assert all(
            mass in list(converted_panel["Mass"]) for mass in (list(necessary_panel["Mass"]))
        )

        # check that correctly formatted panel loads without issue
        converted_panel.to_csv(os.path.join(temp_dir, "converted_panel.csv"), index=False)
        result_panel = panel_utils.convert_panel(os.path.join(temp_dir, "converted_panel.csv"))
        assert list(result_panel.columns) == ["Mass", "Target", "Start", "Stop"]

        # file which is not ionpath or toffy should raise error
        bad_panel = pd.DataFrame({"bad_col1": [], "bad_col2": []})
        bad_panel.to_csv(os.path.join(temp_dir, "bad_panel.csv"), index=False)
        with pytest.raises(ValueError, match="not an Ionpath or toffy structured panel."):
            bad_panel_convert = panel_utils.convert_panel(os.path.join(temp_dir, "bad_panel.csv"))


def mock_panel_conversion(panel_path):
    toffy_panel = pd.DataFrame(
        {"Mass": [69], "Target": ["Calprotectin"], "Start": [68.7], "Stop": [69]}
    )

    return toffy_panel


def test_load_panel(mocker):
    mocker.patch("toffy.panel_utils.convert_panel", mock_panel_conversion)
    toffy_panel = pd.DataFrame(
        {
            "Mass": [69, 71, 89],
            "Target": ["Calprotectin", "Chymase", "Mast Cell Tryptase"],
            "Start": [68.7, 70.7, 88.7],
            "Stop": [69, 71, 89],
        }
    )

    converted_panel = pd.DataFrame(
        {"Mass": [69], "Target": ["Calprotectin"], "Start": [68.7], "Stop": [69]}
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # check that if no existing toffy panel, convert_panel() creates one from original
        panel = panel_utils.load_panel(os.path.join(temp_dir, "test_panel.csv"))
        assert panel.equals(converted_panel)

    with tempfile.TemporaryDirectory() as temp_dir:
        # toffy panel
        toffy_panel.to_csv(os.path.join(temp_dir, "test_panel-toffy.csv"), index=False)

        # check that previously converted panel is loaded
        panel = panel_utils.load_panel(os.path.join(temp_dir, "test_panel-toffy.csv"))
        assert panel.equals(toffy_panel)

        # check that original panel is not read in if converted already exists
        test_utils._make_blank_file(temp_dir, "test_panel.csv")
        panel = panel_utils.load_panel(os.path.join(temp_dir, "test_panel.csv"))
        assert panel.equals(toffy_panel)

        # incorrect formatting with -toffy name should raise an error
        bad_panel = panel.drop(columns=["Mass"])
        bad_panel.to_csv(os.path.join(temp_dir, "bad_panel-toffy.csv"))
        with pytest.raises(ValueError, match="not correctly formatted"):
            panel_utils.load_panel(os.path.join(temp_dir, "bad_panel-toffy.csv"))
