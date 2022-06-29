import pandas as pd
import pytest
import os
import tempfile
import json

from pathlib import Path

from toffy import mph_comp as mph


def create_sample_mph_data(fov, mph_value, total_count, time):
    data = pd.DataFrame([{
        'fov': fov,
        'MPH': mph_value,
        'total_count': total_count,
        'time': time,
    }])
    return data


def test_get_estimated_time():
    bad_path = os.path.join(Path(__file__).parent, "data", "not-a-folder")
    bad_fov = "not-a-fov"

    good_path = os.path.join(Path(__file__).parent, "data", "tissue")
    good_fov = 'fov-1-scan-1'

    # bad directory path should raise an error
    with pytest.raises(ValueError):
        mph.get_estimated_time(bad_path, good_fov)

    # bad fov name data should raise an error
    with pytest.raises(FileNotFoundError):
        mph.get_estimated_time(good_path, bad_fov)

    # bad FOV json file data should raise an error, no frameSize or dwellTimeMillis keys
    bad_data = {"fov": {"not_frameSize": 0, "not_dwellTimeMillis": 0}}
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'fov_name.json'), 'w') as f:
            json.dump(bad_data, f)
        with pytest.raises(KeyError, match="missing one of the necessary keys"):
            mph.get_estimated_time(tmpdir, 'fov_name')

    # test successful time data retrieval
    assert mph.get_estimated_time(good_path, good_fov) == 512


def test_generate_time_ticks():
    example_df = {'cum_total_count': list(range(1, 10)), 'cum_total_time': [5, 9, 11, 12.5, 14, 16, 19, 20, 23]}
    example_df = pd.DataFrame(example_df)
    for i in range(0, len(example_df)):
        example_df['cum_total_time'][i] = example_df['cum_total_time'][i] * 3600 * 1000
        example_df['cum_total_count'][i] = example_df['cum_total_count'][i] * 1000000
    new_ticks = mph.generate_time_ticks(example_df)

    correct_locations = [0, 1, 4, 7]
    correct_labels = [0, 6, 12, 18]

    # test successful new ticks
    assert new_ticks[0] == correct_locations
    assert new_ticks[1] == correct_labels


def test_compute_mph_metrics():
    bin_file_path = os.path.join(Path(__file__).parent, "data", "tissue")
    fov_name = 'fov-1-scan-1'

    with tempfile.TemporaryDirectory() as tmpdir:

        # invalid fov name should raise an error
        with pytest.raises(FileNotFoundError):
            mph.compute_mph_metrics(bin_file_path, tmpdir, "not-a-fov")

        # test successful data retrieval and csv output
        mph.compute_mph_metrics(bin_file_path, tmpdir, fov_name)
        csv_path = os.path.join(tmpdir, fov_name + '-mph_pulse.csv')
        assert os.path.exists(csv_path)

        # check the csv data is correct
        mph_data = create_sample_mph_data(fov_name, 3404, 72060, 512)
        csv_data = pd.read_csv(csv_path)
        assert csv_data.equals(mph_data)


def test_combine_mph_metrics():
    bad_path = os.path.join(Path(__file__).parent, "data", "not-a-folder")

    # bad directory path should raise an error
    with pytest.raises(ValueError):
        mph.combine_mph_metrics(bad_path)

    data1 = create_sample_mph_data(fov='fov-1', mph_value=1000, total_count=50000, time=500)
    data2 = create_sample_mph_data(fov='fov-2', mph_value=2000, total_count=70000, time=500)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = tmpdir

        data1.to_csv(os.path.join(csv_path, 'fov-1-scan-1-mph_pulse.csv'), index=False)
        data2.to_csv(os.path.join(csv_path, 'fov-2-scan-1-mph_pulse.csv'), index=False)

        combined_data = pd.concat([data1, data2], axis=0, ignore_index=True)
        combined_data['cum_total_count'] = [50000, 120000]
        combined_data['cum_total_time'] = [500, 1000]

        # test successful data retrieval and csv output
        mph.combine_mph_metrics(csv_path)
        combined_csv_path = os.path.join(csv_path, 'mph_pulse_combined.csv')
        csv_data = pd.read_csv(combined_csv_path)
        assert os.path.exists(combined_csv_path)
        assert csv_data.equals(combined_data)


def test_visualize_mph():
    bad_path = os.path.join(Path(__file__).parent, "data", "not-a-folder")
    mph_data = pd.DataFrame({
            'MPH': [2222, 3800],
            'cum_total_count': [72060, 146859],
            'cum_total_time': [512, 1024],
        }, index=[0, 1])

    # bad output directory path should raise an error
    with pytest.raises(ValueError):
        mph.visualize_mph(mph_data, bad_path, regression=False)

    with tempfile.TemporaryDirectory() as temp_dir:
        # test for saving to directory
        mph.visualize_mph(mph_data, out_dir=temp_dir, regression=True)
        assert os.path.exists(os.path.join(temp_dir, 'fov_vs_mph.jpg'))
