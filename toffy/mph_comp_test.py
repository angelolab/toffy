import pandas as pd
import pytest
import os
import tempfile
import json

from toffy import mph_comp as mph


def get_estimated_time():
    bad_path = os.path.join("data", "not-a-folder")
    bad_fov = "not-a-fov"

    good_path = os.path.join("data", "tissue")
    good_fov = 'fov-1-scan-1'

    # bad directory path should raise an error
    with pytest.raises(ValueError):
        mph.get_estimated_time(bad_path, good_fov)

    # bad fov name data should raise an error
    with pytest.raises(FileNotFoundError):
        mph.get_estimated_time(good_path, bad_fov)

    # bad FOV json file data should raise an error, no frameSize or dwellTimeMillis keys
    bad_data = {'fov': {'not_frameSize': 0, 'not_dwellTimeMillis': 0}}
    temp_json = tempfile.NamedTemporaryFile(mode="w", suffix='fov_name.json', delete=False)
    temp_json.write(json.dumps(bad_data))
    temp_dir = tempfile.gettempdir()
    with pytest.raises(KeyError, match="missing one of the necessary keys"):
        mph.get_estimated_time(temp_dir, 'fov_name')

    # test successful time data retrieval
    assert mph.get_estimated_time(good_path, good_fov) == 512


def test_compute_mph_metrics():
    bin_file_path = os.path.join("data", "tissue")
    fov_name = 'fov-1-scan-1'
    target_name = 'CD8'
    start_mass = -0.3
    stop_mass = 0.0

    # bad directory path should raise an error
    bad_path = os.path.join("data", "not-a-folder")
    with pytest.raises(ValueError):
        mph.compute_mph_metrics(bad_path, bin_file_path, fov_name,
                                target_name, start_mass, stop_mass)

    # invalid fov name should raise an error
    with pytest.raises(FileNotFoundError):
        mph.compute_mph_metrics(bin_file_path, bin_file_path, "not-a-fov",
                                target_name, start_mass, stop_mass)

    # invalid target name should raise an error
    with pytest.raises(ValueError, match="target name is invalid"):
        mph.compute_mph_metrics(bin_file_path, bin_file_path, fov_name,
                                "not-a-target", start_mass, stop_mass)

    # test successful data retrieval and csv output
    mph.compute_mph_metrics(bin_file_path, bin_file_path, fov_name,
                            target_name, start_mass, stop_mass)
    csv_path = os.path.join(bin_file_path, fov_name + '-pulse_height.csv')
    assert os.path.exists(csv_path)

    # check the csv data is correct
    mph_data = pd.DataFrame([{
            'fov': fov_name,
            'MPH': 2222,
            'total_count': 72060,
            'time': 512,
        }])
    csv_data = pd.read_csv(csv_path)
    assert csv_data.equals(mph_data)

    os.remove(os.path.join(bin_file_path, 'fov-1-scan-1-pulse_height.csv'))


def test_combine_mph_metrics():
    csv_path = os.path.join("data", "tissue")
    bad_path = os.path.join("data", "not-a-folder")

    # bad directory path should raise an error
    with pytest.raises(ValueError):
        mph.combine_mph_metrics(bad_path)

    data1 = pd.DataFrame([{
        'fov': 'fov-1-scan-1',
        'MPH': 2222,
        'total_count': 72060,
        'time': 512,
    }])
    data2 = pd.DataFrame([{
        'fov': 'fov-2-scan-1',
        'MPH': 3800,
        'total_count': 74799,
        'time': 512,
    }])

    data1.to_csv(os.path.join(csv_path, 'fov-1-scan-1-pulse_height.csv'), index=False)
    data2.to_csv(os.path.join(csv_path, 'fov-2-scan-1-pulse_height.csv'), index=False)

    combined_data = pd.DataFrame({
        'fov': ['fov-1-scan-1', 'fov-2-scan-1'],
        'MPH': [2222, 3800],
        'total_count': [72060, 74799],
        'time': [512, 512],
        'cum_total_count': [72060, 146859],
        'cum_total_time': [512, 1024],
        }, index=[0, 1])

    # test successful data retrieval and csv output
    mph_data = mph.combine_mph_metrics(csv_path, return_data=True)
    combined_csv_path = os.path.join(csv_path, 'total_count_vs_mph_data.csv')
    csv_data = pd.read_csv(combined_csv_path)
    assert os.path.exists(combined_csv_path)
    assert csv_data.equals(combined_data)

    os.remove(combined_csv_path)
    os.remove(os.path.join(csv_path, 'fov-1-scan-1-pulse_height.csv'))
    os.remove(os.path.join(csv_path, 'fov-2-scan-1-pulse_height.csv'))


def test_visualize_mph():
    bad_path = os.path.join("data", "not-a-folder")
    mph_data = pd.DataFrame({
            'MPH': [2222, 3800],
            'cum_total_count': [72060, 146859],
            'cum_total_time': [512, 1024],
        }, index=[0, 1])

    # bad output directory path should raise an error
    with pytest.raises(ValueError):
        mph.visualize_mph(mph_data, False, bad_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        # test without saving
        mph.visualize_mph(mph_data, True)
        assert not os.path.exists(os.path.join(temp_dir, 'fov_vs_mph.jpg'))

        # test with saving
        mph.visualize_mph(mph_data, True, save_dir=temp_dir)
        assert os.path.exists(os.path.join(temp_dir, 'fov_vs_mph.jpg'))
