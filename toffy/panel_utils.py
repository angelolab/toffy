import os
import pandas as pd

from pathlib import Path

from ark.utils import io_utils


def convert_panel(panel_path):
    """
    Converts the panel retrieved from ionpath into a necessary toffy format,
    also adds necessary channels for analysis
    Args:
        panel_path (str): direct path to panel file
    Returns:
        panel (pd.DataFrame): detailing the Mass, Target, Start, and Stop values
        also saves the above to file with name formatted "panel_name-toffy.csv"
    """

    # retrieve the panel name and directory it is contained in
    panel_name = os.path.basename(panel_path).split('.')[0]
    panel_dir = os.path.dirname(panel_path)

    # read in the example_panel_file to have necessary additional targets
    example_panel = pd.read_csv(os.path.join(Path(__file__).parent.parent, 'files',
                                             'example_panel_file.csv'), index_col=False)

    # read in metadata contained in the first few lines
    with open(panel_path) as r:
        lines = r.readlines()
        for index, line in enumerate(lines):
            # line containing the column names we want
            if len(line.split(',')) == 10:
                col_index = index
                break
        r.close()

    # load in the panel info to df
    with open(panel_path) as r:
        headers = [next(r) for i in list(range(col_index+1))]
        cols = headers[col_index].split(',')
        panel = pd.read_csv(r, sep=',', names=cols, index_col=False)
        r.close()

    # retrieve column names, and original mass / target values
    panel.columns = [panel.columns.str.replace('"', '')]
    panel.columns = panel.columns.get_level_values(0)
    toffy_panel = panel[['Mass', 'Target']].copy()

    mass_start = []
    mass_stop = []

    for i, row in toffy_panel.iterrows():
        mass = row['Mass']
        # check for different targets on same mass
        if mass in mass_stop:
            dup_index = mass_stop.index(mass)
            # concatenate target names
            target_name = toffy_panel['Target'][dup_index] + '_' + row['Target']
            toffy_panel.loc[toffy_panel['Mass'] == mass, 'Target'] = target_name
        else:
            mass_start.append(mass - 0.3)
            mass_stop.append(mass)

    # get rid of one of the duplicate mass rows
    toffy_panel = toffy_panel.drop_duplicates(subset=['Mass'], keep='first')

    # edit panel to add Start and Stop values
    toffy_panel['Start'] = mass_start
    toffy_panel['Stop'] = mass_stop

    # add example panel targets to new toffy panel, removing any duplicates
    toffy_panel = pd.concat([toffy_panel, example_panel], ignore_index=True)
    toffy_panel = toffy_panel.drop_duplicates(subset=['Mass'], keep='first')

    toffy_panel.to_csv(os.path.join(panel_dir, panel_name + '-toffy.csv'), index=False)

    return toffy_panel


def load_panel(panel_path):
    """Loads in the toffy panel data, calls convert_panel() if necessary
    Args:
        panel_path (str): direct path to panel file
    Returns:
        panel (pd.DataFrame): toffy formatted panel data
    """

    # read in the provided panel info
    panel_name = os.path.basename(panel_path).split('.')[0]
    panel_dir = os.path.dirname(panel_path)

    # if panel path points to toffy panel, load it
    if '-toffy' in panel_name:
        toffy_panel = pd.read_csv(panel_path, index_col=False)
    else:
        # check if toffy panel exists in panel_dir
        files = io_utils.list_files(panel_dir, substrs=panel_name)
        if panel_name + '-toffy.csv' in files:
            converted_panel_path = os.path.join(panel_dir, panel_name + '-toffy.csv')
            toffy_panel = pd.read_csv(converted_panel_path, index_col=False)
        # if no toffy panel, create one
        else:
            toffy_panel = convert_panel(panel_path)

    return toffy_panel
