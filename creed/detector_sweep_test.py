import os
import pytest
import tempfile

from creed import detector_sweep


def test_parse_sweep_parameters():

    voltage = '2700'
    date = '2022-01-16'
    time = '23-06-24'
    name = 'Detector_' + voltage + 'v_' + date + '_' + time

    sweep_file = detector_sweep.parse_sweep_parameters(name)

    assert sweep_file.date == date
    assert sweep_file.time == time.replace('-', '')
    assert sweep_file.voltage == int(voltage)


def test_find_detector_sweep_folders():
    with tempfile.TemporaryDirectory() as temp_dir:

        # create sweep folders with default pattern
        date = '2022-01-16'
        times = ['23-07-{}'.format(i) for i in range(10, 60, 5)]
        voltages = [2700 + 25 * i for i in range(10)]
        sweeps = ['Detector_' + str(voltages[i]) + 'v_' + date + '_' + times[i] for i in range(10)]

        # create sweep folders which match only a subset of conditions
        bad_date = 'Detector_' + str(voltages[2]) + 'v_' + '2022-01-17_' + times[2]
        bad_time = 'Detector_' + str(voltages[2]) + 'v_' + date + '_24-07-15'

        # create sweep folder with wrong voltage
        bad_voltage = 'Detector_' + '2715v_' + date + '_' + times[2]

        for folder in sweeps + [bad_date] + [bad_time]:
            os.makedirs(os.path.join(temp_dir, folder))

        predicted_folders = detector_sweep.find_detector_sweep_folders(temp_dir, sweeps[0],
                                                                       sweeps[-1])
        assert set(predicted_folders) == set(sweeps)

        # bad voltage should raise an error
        os.makedirs(os.path.join(temp_dir, bad_voltage))
        with pytest.raises(ValueError):
            _ = detector_sweep.find_detector_sweep_folders(temp_dir, sweeps[0], sweeps[-1])

        # detector sweep with different start and end dates should raise and error
        with pytest.raises(ValueError):
            _ = detector_sweep.find_detector_sweep_folders(temp_dir, sweeps[0], bad_date)
