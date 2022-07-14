import os.path

from mibi_bin_tools import io_utils
from toffy import json_utils


def check_detector_voltage(run_dir):
    """ Check all FOVs in a run to determine whether the detector voltage stays constant
    Args:
        run_dir(string): path to directory containing json files of all fovs in the run
    Return:
        raise error if changes in voltage were found between fovs
    """

    fovs = io_utils.remove_file_extensions(io_utils.list_files(run_dir, substrs='.bin'))
    changes_in_voltage = []

    for i, fov in enumerate(fovs):
        fov_data = json_utils.read_json_file(os.path.join(run_dir, fov+'.json'))
        for j in range(0, len(fov_data['hvDac'])):
            if fov_data['hvDac'][j]['name'] == 'Detector':
                index = j
                break
        fov_voltage = fov_data['hvDac'][index]['currentSetPoint']
        if i == 0:
            voltage_level = fov_voltage
        elif fov_voltage != voltage_level:
            changes_in_voltage.append({fovs[i - 1]: voltage_level, fovs[i]: fov_voltage})
            voltage_level = fov_voltage

    if changes_in_voltage:
        raise ValueError(f'Changes in detector voltage were found during '
                         f'the run: {changes_in_voltage}')
