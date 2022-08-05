import os
import csv
import pandas as pd

from ark.utils import io_utils


def convert_panel(panel_path):
    panel_name = io_utils.remove_file_extensions([os.path.basename(panel_path)])
    panel_dir = os.path.dirname(panel_path)

    with open(panel_path) as r:
        headers = [next(r) for i in list(range(11))]
        cols = headers[10].split(',')
        panel = pd.read_csv(r, sep=',', names=cols, index_col=False)

    # retrieve original mass and target values
    panel.columns = [panel.columns.str.replace('"', '')]
    toffy_panel = panel[['Mass', 'Target']]

    # edit panel
    for mass in toffy_panel['Mass']:
        print(1)

    toffy_panel.to_csv(os.path.join(panel_dir, panel_name + '-toffy.csv'))


path = os.path.join('..', 'files', 'Panel106.csv')
convert_panel(path)


def load_panel(panel_path):
    panel_name = io_utils.remove_file_extensions(os.path.basename(panel_path))
    panel_dir = os.path.dirname(panel_path)

    # if panel path points to toffy panel, read in
    if '-toffy' in panel_name:
        toffy_panel = pd.read_csv(panel_path)
    else:
        files = io_utils.remove_file_extensions(io_utils.list_files(panel_dir, substrs=panel_name))
        # check if toffy panel exists in panel_dir
        if panel_name + '-toffy.csv' in files:
            converted_panel_path = os.path.join(panel_dir, panel_name + '-toffy.csv')
            toffy_panel = pd.read_csv(converted_panel_path)
        # if no toffy panel, create one
        else:
            toffy_panel = convert_panel(panel_path)

    return toffy_panel





