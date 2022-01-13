from ark.utils import io_utils


def extract_sweep_parameters(sweep_name):
    _, voltage, date, time = sweep_name.split('_')
    voltage = int(voltage[:-1])
    time = time.replace('-', '')

    return voltage, date, time


def find_detector_sweep_folders(data_dir, first_fov, last_fov, sweep_step=25):
    first_voltage, first_date, first_time = extract_sweep_parameters(first_fov)
    last_voltage, last_date, last_time = extract_sweep_parameters(last_fov)

    if last_date != first_date:
        raise ValueError("The day of the first fov {} is not the same as"
                         "the day of the last fov {}".format(first_date, last_date))

    # get all folders with matching names
    potential_sweeps = io_utils.list_folders(data_dir, 'Detector')
    potential_sweeps = [folder for folder in potential_sweeps if first_date in folder]

    sweep_voltages = range(first_voltage + sweep_step, last_voltage, sweep_step)

    final_folder_list = [first_fov]
    for sweep_voltage in sweep_voltages:
        matches = [sweep for sweep in potential_sweeps if str(sweep_voltage) in sweep]

        if len(matches) == 0:
            raise ValueError('No matching sweep found for voltage {}'.format(sweep_voltage))
        elif len(matches) == 1:
            final_folder_list.append(matches[0])
        else:
            # determine which of the matches has a time between first and last fov
            candidates = []
            for potential_match in matches:
                _, _, potential_time = extract_sweep_parameters(potential_match)
                if first_time < potential_time < last_time:
                    candidates.append(potential_match)
            if len(candidates) == 1:
                final_folder_list.append(candidates[0])
            else:
                raise ValueError('Did not find one-to-one match for FOVs {}'.format(candidates))

    final_folder_list.append(last_fov)

    return final_folder_list
