import os

import pandas as pd
from alpineer import io_utils

necessary_masses = pd.DataFrame(
    {
        "Mass": [
            39,
            48,
            56,
            69,
            71,
            89,
            113,
            115,
            117,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            164,
            165,
            166,
            167,
            168,
            169,
            170,
            171,
            172,
            173,
            174,
            175,
            176,
            197,
        ],
        "Target": [
            "chan_39",
            "chan_48",
            "Fe",
            "chan_69",
            "chan_71",
            "chan_89",
            "chan_113",
            "chan_115",
            "Noodle",
            "chan_141",
            "chan_142",
            "chan_143",
            "chan_144",
            "chan_145",
            "chan_146",
            "chan_147",
            "chan_148",
            "chan_149",
            "chan_150",
            "chan_151",
            "chan_152",
            "chan_153",
            "chan_154",
            "chan_155",
            "chan_156",
            "chan_157",
            "chan_158",
            "chan_159",
            "chan_160",
            "chan_161",
            "chan_162",
            "chan_163",
            "chan_164",
            "chan_165",
            "chan_166",
            "chan_167",
            "chan_168",
            "chan_169",
            "chan_170",
            "chan_171",
            "chan_172",
            "chan_173",
            "chan_174",
            "chan_175",
            "chan_176",
            "Au",
        ],
    }
)


def merge_duplicate_masses(panel):
    """Check a panel df for duplicate mass values and return a unique mass panel with the
        target names combined
    Args:
        panel (pd.DataFrame): panel dataframe with columns Mass and Target
    Returns:
        pd.DataFrame with no duplicate masses
    """

    # find the mass and target values of duplicate
    duplicates = panel[panel["Mass"].duplicated(keep=False)]

    # combine target names
    for mass in duplicates["Mass"].unique():
        duplicate_targets = panel.loc[panel["Mass"] == mass, "Target"]
        target_name = "".join(str(name) + "_" for name in duplicate_targets)[:-1]
        panel.loc[panel["Mass"] == mass, "Target"] = target_name

    # only keep one row for each mass value
    unique_panel = panel.drop_duplicates(subset=["Mass"], keep="first")

    return unique_panel


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
    panel_name = os.path.basename(os.path.splitext(panel_path)[0])
    panel_dir = os.path.dirname(panel_path)

    # read in metadata contained in the first few lines
    col_index = 0
    with open(panel_path, encoding="utf-8") as r:
        lines = r.readlines()
        for index, line in enumerate(lines):
            # line containing the column names we want
            if len(line.split(",")) == 10:
                col_index = index
                break
        r.close()

    # load in the panel info to df
    with open(panel_path, encoding="utf-8-sig") as r:
        headers = [next(r) for i in list(range(col_index + 1))]
        cols = headers[col_index].split(",")
        panel = pd.read_csv(r, sep=",", names=cols, index_col=False)
        r.close()

    panel.columns = [panel.columns.str.replace('"', "")]
    panel.columns = panel.columns.get_level_values(0)

    # check for already correctly formatted panel, return as is
    if list(panel.columns) == ["Mass", "Target", "Start", "Stop\n"]:
        print(f"{panel_name}.csv has the correct toffy format. Loading in panel data.")
        panel.columns = ["Mass", "Target", "Start", "Stop"]
        return panel
    # if not ionpath panel, raise error
    elif list(panel.columns) != [
        "ID (Lot)",
        "Target",
        "Clone",
        "Mass",
        "Element",
        "Manufactured",
        "Stock",
        "Titer",
        "Volume (μL)",
        "Staining Batch\n",
    ]:
        raise ValueError(f"{panel_name}.csv is not an Ionpath or toffy structured panel.")

    # retrieve original mass / target values
    toffy_panel = panel[["Mass", "Target"]].copy()

    # include only the unique provided masses with concatenated target names
    toffy_panel = merge_duplicate_masses(toffy_panel)

    # add necessary panel masses to the toffy panel, removing any duplicates
    toffy_panel = pd.concat([toffy_panel, necessary_masses], ignore_index=True)
    toffy_panel = toffy_panel.drop_duplicates(subset=["Mass"], keep="first")

    # edit panel to add Start and Stop values
    toffy_panel["Start"] = toffy_panel["Mass"].copy() - 0.3
    toffy_panel["Stop"] = toffy_panel["Mass"].copy()

    # sort data by mass
    toffy_panel = toffy_panel.sort_values(by=["Mass"])

    toffy_panel.to_csv(os.path.join(panel_dir, panel_name + "-toffy.csv"), index=False)
    print(
        f"{panel_name}.csv does not have the correct toffy format. "
        f"Creating {panel_name}-toffy.csv and loading in panel data."
    )

    return toffy_panel


def load_panel(panel_path):
    """Loads in the toffy panel data, calls convert_panel() if necessary
    Args:
        panel_path (str): direct path to panel file
    Returns:
        panel (pd.DataFrame): toffy formatted panel data
    """

    # read in the provided panel info
    panel_name = os.path.basename(panel_path).split(".")[0]
    panel_dir = os.path.dirname(panel_path)

    # if panel path points to toffy panel, load it
    if "-toffy" in panel_name:
        toffy_panel = pd.read_csv(panel_path, index_col=False)
        if list(toffy_panel.columns) != ["Mass", "Target", "Start", "Stop"]:
            raise ValueError(
                f"{panel_name}.csv is not correctly formatted. Please remove "
                "'-toffy' from the file name."
            )
        else:
            print(f"{panel_name}.csv is in the correct toffy format. Loading in panel data.")
    else:
        # check if toffy panel exists in panel_dir and load it
        files = io_utils.list_files(panel_dir, substrs=panel_name)
        if panel_name + "-toffy.csv" in files:
            converted_panel_path = os.path.join(panel_dir, panel_name + "-toffy.csv")
            toffy_panel = pd.read_csv(converted_panel_path, index_col=False)
            print(
                f"Detected {panel_name}-toffy.csv in {panel_dir}. "
                "Loading in toffy formatted panel data."
            )
        # if no toffy panel, create one
        else:
            toffy_panel = convert_panel(panel_path)

    return toffy_panel
