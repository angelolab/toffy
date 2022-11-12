from dataclasses import dataclass

from tmi import io_utils, misc_utils


@dataclass
class VoltageSweepFile:
    voltage: int
    date: str
    time: str


def parse_sweep_parameters(sweep_name):
    """Performs string manipulations to get the sweep parameters from an FOV folder

    Args:
        sweep_name: the name of a folder containing a single FOV of a detector sweep

    Returns
        tuple: the extracted voltage, date, and time"""

    _, voltage, date, time = sweep_name.split('_')
    voltage = int(voltage[:-1])
    time = time.replace('-', '')

    return VoltageSweepFile(voltage, date, time)


def find_detector_sweep_folders(data_dir, first_fov, last_fov, sweep_step=25):
    """Generates the names of all folders from a single detector sweep

    Args:
        data_dir: directory where runs are saved
        first_fov: the name of the first FOV from the detector sweep
        last_fov: the name of the last FOV from the detector sweep
        sweep_step: the voltage increase between FOVs in the detector sweep

    Returns:
        list: the names of all detector sweep folders"""

    first_sweep = parse_sweep_parameters(first_fov)
    last_sweep = parse_sweep_parameters(last_fov)

    if first_sweep.date != last_sweep.date:
        raise ValueError("The day of the first fov {} is not the same as"
                         "the day of the last fov {}".format(first_sweep.date, last_sweep.date))

    # get all folders with matching date
    potential_sweeps = io_utils.list_folders(data_dir, 'Detector')
    potential_sweeps = [sweep for sweep in potential_sweeps if first_sweep.date in sweep]

    # only keep folders created between start and end time
    potential_sweeps = [sweep for sweep in potential_sweeps if
                        first_sweep.time < parse_sweep_parameters(sweep).time < last_sweep.time]

    pred_sweep_voltages = range(first_sweep.voltage + sweep_step, last_sweep.voltage, sweep_step)
    obs_sweep_voltages = [parse_sweep_parameters(sweep).voltage for sweep in potential_sweeps]

    misc_utils.verify_same_elements(predicted_voltages=pred_sweep_voltages,
                                    observed_voltages=obs_sweep_voltages)

    final_folder_list = [first_fov] + potential_sweeps + [last_fov]

    return final_folder_list
