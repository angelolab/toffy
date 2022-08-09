import os
import csv
import tempfile
import pandas as pd

from toffy import panel_utils
from ark.utils import test_utils


def test_convert_panel():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_panel = pd.DataFrame({
            'ID (Lot)': [1352, 1352, 1350, 1351],
            'Target': ['Calprotectin', 'Duplicate', 'Chymase', 'Mast Cell Tryptase'],
            'Clone': ['MAC387', 'MAC387', 'EPR13136', 'EPR9522'],
            'Mass': [69, 69, 71, 89],
            'Element': ['Ga', 'Ga', 'Ga', 'Y'],
            'Manufacture': ['7/20/20', '7/20/20', '7/20/20', '5/5/21'],
            'Stock': [200, 200, 200, 200],
            'Titer': [0.125, 0.125, 0.125, 0.25],
            'Volume': [0, 0, 0, 0],
            'Staining Batch': [1, 1, 1, 1]
        })
        test_panel.to_csv(os.path.join(temp_dir, 'test_panel.csv'), index=False)

        # add metadata
        with open(os.path.join(temp_dir, 'test_panel.csv'), newline='') as f:
            r = csv.reader(f)
            data = [line for line in r]
        with open(os.path.join(temp_dir, 'test_panel.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            metadata = '"Panel ID",106\r\n"Panel Name","DCIS_followup"' \
                       '\r\n"Manufacture Date","2021-08-25"\r\n"Description",""\r\n"Batch",1' \
                       '\r\n"Total Volume (μL)",0\r\n"Antibody Volume (μL)",0\r\n' \
                       '"Buffer Volume (μL)",0'
            w.writerows([metadata])
            w.writerows(data)

        # test successful new panel creation
        panel_utils.convert_panel(os.path.join(temp_dir, 'test_panel.csv'))
        assert os.path.exists(os.path.join(temp_dir, 'test_panel-toffy.csv'))

        converted_panel = pd.read_csv(os.path.join(temp_dir, 'test_panel-toffy.csv'))
        necessary_panel = pd.read_csv(os.path.join('..', 'files', 'example_panel_file.csv'))

        # check toffy panel structure
        assert list(converted_panel.columns) == ['Mass', 'Target', 'Start', 'Stop']

        # check chan_71 is not duplicated
        assert ['chan_71', 'Duplicate'] not in list(converted_panel['Target'])

        # check concatenated target names for same mass
        assert 'Calprotectin_Duplicate' in list(converted_panel['Target'])

        # check for unique mass values
        assert len(list(converted_panel['Mass'])) == len(set(converted_panel['Mass']))

        # check for all necessary masses
        assert all(mass in list(converted_panel['Mass'])
                   for mass in (list(necessary_panel['Mass'])))


def mock_panel_conversion(panel_path):
    toffy_panel = pd.DataFrame({
        'Target': ['Calprotectin', 'Chymase', 'Mast Cell Tryptase'],
        'Mass': [69, 71, 89],
        'Start': [68.7, 70.7, 88.7],
        'Stop': [69, 71, 89]
    })

    return toffy_panel


def test_load_panel(mocker):
    mocker.patch('toffy.panel_utils.convert_panel', mock_panel_conversion)
    toffy_panel = pd.DataFrame({
        'Target': ['Calprotectin', 'Chymase', 'Mast Cell Tryptase'],
        'Mass': [69, 71, 89],
        'Start': [68.7, 70.7, 88.7],
        'Stop': [69, 71, 89]
    })

    with tempfile.TemporaryDirectory() as temp_dir:
        # check that if no existing toffy panel, create one from original
        panel = panel_utils.load_panel(os.path.join(temp_dir, 'test_panel.csv'))
        assert panel.equals(toffy_panel)

    with tempfile.TemporaryDirectory() as temp_dir:
        # toffy panel
        toffy_panel.to_csv(os.path.join(temp_dir, 'test_panel-toffy.csv'), index=False)

        # check that previously converted panel is loaded
        panel = panel_utils.load_panel(os.path.join(temp_dir, 'test_panel-toffy.csv'))
        assert panel.equals(toffy_panel)

        # check that original panel is not read in if converted already exists
        test_utils._make_blank_file(temp_dir, 'test_panel.csv')
        panel = panel_utils.load_panel(os.path.join(temp_dir, 'test_panel.csv'))
        assert panel.equals(toffy_panel)
