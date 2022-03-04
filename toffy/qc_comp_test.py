import numpy as np
import pandas as pd
from requests.exceptions import HTTPError
import xarray as xr

from toffy.mibitracker_utils import MibiRequests
from toffy import qc_comp
import ark.utils.io_utils as io_utils
import ark.utils.misc_utils as misc_utils
import ark.utils.test_utils as test_utils

import os
from pathlib import Path
import pytest
import tempfile

parametrize = pytest.mark.parametrize


RUN_POINT_NAMES = ['Point%d' % i for i in range(1, 13)]
RUN_POINT_IDS = list(range(661, 673))

# NOTE: all fovs and all channels will be tested in the example_qc_metric_eval notebook test
FOVS_CHANS_TEST_MIBI = [
    (None, ['CCL8', 'CD11b'], None, RUN_POINT_NAMES, RUN_POINT_IDS),
    (None, ['CCL8', 'CD11b'], "TIFs", RUN_POINT_NAMES, RUN_POINT_IDS),
    (['Point1'], None, None, RUN_POINT_NAMES[0:1], RUN_POINT_IDS[0:1]),
    (['Point1'], None, "TIFs", RUN_POINT_NAMES[0:1], RUN_POINT_IDS[0:1]),
    (['Point1'], ['CCL8', 'CD11b'], None, RUN_POINT_NAMES[0:1], RUN_POINT_IDS[0:1]),
    (['Point1'], ['CCL8', 'CD11b'], "TIFs", RUN_POINT_NAMES[0:1], RUN_POINT_IDS[0:1])
]


FOVS_CHANS_TEST_QC = [
    (None, None, False),
    (None, None, True),
    (['fov0', 'fov1'], None, False),
    (['fov0', 'fov1'], None, True),
    (None, ['chan0', 'chan1'], False),
    (None, ['chan0', 'chan1'], True),
    (['fov0', 'fov1'], ['chan0', 'chan1'], False),
    (['fov0', 'fov1'], ['chan0', 'chan1'], True)
]

MIBITRACKER_EMAIL = 'qc.mibi@gmail.com'
MIBITRACKER_PASSWORD = 'The_MIBI_Is_Down_Again1!?'
MIBITRACKER_RUN_NAME = '191008_JG85b'
MIBITRACKER_RUN_LABEL = 'JG85_Run2'


def test_create_mibitracker_request_helper():
    # error check: bad email and/or password provided
    mr = qc_comp.create_mibitracker_request_helper('bad_email', 'bad_password')
    assert mr is None

    # test creation works (just test the correct type returned)
    mr = qc_comp.create_mibitracker_request_helper(MIBITRACKER_EMAIL, MIBITRACKER_PASSWORD)
    assert type(mr) == MibiRequests


@pytest.mark.parametrize(
    "test_fovs,test_chans,test_sub_folder,actual_points,actual_ids",
    FOVS_CHANS_TEST_MIBI
)
def test_download_mibitracker_data(test_fovs, test_chans, test_sub_folder,
                                   actual_points, actual_ids):
    with tempfile.TemporaryDirectory() as temp_dir:
        # error check: bad base_dir provided
        with pytest.raises(FileNotFoundError):
            qc_comp.download_mibitracker_data('', '', '', '', 'bad_base_dir', '', '')

        # error check: bad run_name and/or run_label provided
        with pytest.raises(ValueError):
            qc_comp.download_mibitracker_data(
                MIBITRACKER_EMAIL, MIBITRACKER_PASSWORD, 'bad_run_name', 'bad_run_label',
                temp_dir, '', ''
            )

        # bad fovs provided
        with pytest.raises(ValueError):
            qc_comp.download_mibitracker_data(
                MIBITRACKER_EMAIL, MIBITRACKER_PASSWORD,
                MIBITRACKER_RUN_NAME, MIBITRACKER_RUN_LABEL,
                temp_dir, '', '', fovs=['Point0', 'Point1']
            )

        # bad channels provided
        with pytest.raises(ValueError):
            qc_comp.download_mibitracker_data(
                MIBITRACKER_EMAIL, MIBITRACKER_PASSWORD,
                MIBITRACKER_RUN_NAME, MIBITRACKER_RUN_LABEL,
                temp_dir, '', '', channels=['B', 'C']
            )

        # ensure test to remove tiff_dir if it already exists runs
        os.mkdir(os.path.join(temp_dir, 'sample_tiff_dir'))

        # error check: tiff_dir that already exists provided with overwrite_tiff_dir=False
        with pytest.raises(ValueError):
            qc_comp.download_mibitracker_data(
                MIBITRACKER_EMAIL, MIBITRACKER_PASSWORD,
                MIBITRACKER_RUN_NAME, MIBITRACKER_RUN_LABEL,
                temp_dir, 'sample_tiff_dir', overwrite_tiff_dir=False,
                img_sub_folder=test_sub_folder, fovs=test_fovs, channels=test_chans
            )

        # run the data
        run_order = qc_comp.download_mibitracker_data(
            MIBITRACKER_EMAIL, MIBITRACKER_PASSWORD,
            MIBITRACKER_RUN_NAME, MIBITRACKER_RUN_LABEL,
            temp_dir, 'sample_tiff_dir', overwrite_tiff_dir=True,
            img_sub_folder=test_sub_folder, fovs=test_fovs, channels=test_chans
        )

        # for testing purposes, set test_fovs and test_chans to all fovs and channels
        # if they're set to None
        if test_fovs is None:
            test_fovs = ['Point%d' % i for i in np.arange(1, 13)]

        if test_chans is None:
            test_chans = [
                'CD115', 'C', 'Au', 'CCL8', 'CD11c', 'Ca', 'Background',
                'CD11b', 'CD192', 'CD19', 'CD206', 'CD25', 'CD4', 'CD45.1',
                'CD3', 'CD31', 'CD49b', 'CD68', 'CD45.2', 'FceRI', 'DNA', 'CD8',
                'F4-80', 'Fe', 'IL-1B', 'Ly-6C', 'FRB', 'Lyve1', 'Ly-6G', 'MHCII',
                'Na', 'Si', 'SMA', 'P', 'Ta', 'TREM2'
            ]

        # set the sub folder to a blank string if None
        if test_sub_folder is None:
            test_sub_folder = ""

        # get the contents of tiff_dir
        tiff_dir_contents = os.listdir(os.path.join(temp_dir, 'sample_tiff_dir'))

        # assert all the fovs are contained in the dir
        tiff_dir_fovs = [d for d in tiff_dir_contents if
                         os.path.isdir(os.path.join(temp_dir, 'sample_tiff_dir', d))]
        misc_utils.verify_same_elements(
            created_fov_dirs=tiff_dir_fovs,
            provided_fov_dirs=test_fovs
        )

        # assert for each fov the channels created are correct
        for fov in tiff_dir_fovs:
            # list all the files in the fov folder (and sub folder)
            # remove file extensions so raw channel names are extracted
            channel_files = io_utils.remove_file_extensions(os.listdir(
                os.path.join(temp_dir, 'sample_tiff_dir', fov, test_sub_folder)
            ))

            # assert the channel names are the same
            misc_utils.verify_same_elements(
                create_channels=channel_files,
                provided_channels=test_chans
            )

        # assert that the run order created is correct for both points and ids
        run_fov_names = [ro[0] for ro in run_order]
        run_fov_ids = [ro[1] for ro in run_order]

        assert run_fov_names == actual_points
        assert run_fov_ids == actual_ids


def test_compute_nonzero_mean_intensity():
    sample_img_arr = np.array([[0, 1, 2], [3, 0, 0], [0, 4, 5]])
    sample_nonzero_mean = qc_comp.compute_nonzero_mean_intensity(sample_img_arr)
    assert sample_nonzero_mean == 3


def test_compute_total_intensity():
    sample_img_arr = np.array([[0, 1, 2], [3, 0, 0], [0, 4, 5]])
    sample_total_intensity = qc_comp.compute_total_intensity(sample_img_arr)
    assert sample_total_intensity == 15


def test_compute_99_9_intensity():
    sample_img_arr = np.array([[0, 1, 2], [3, 0, 0], [0, 4, 5]])
    sample_99_9_intensity = qc_comp.compute_99_9_intensity(sample_img_arr)
    assert np.allclose(sample_99_9_intensity, 5, rtol=1e-02)


# NOTE: we don't need to test iteration over multiple FOVs because
# test_compute_qc_metrics computes on 1 FOV at a time
@parametrize("gaussian_blur", [False, True])
@parametrize("bin_file_folder, fovs",
             [('moly', ['fov-1-scan-1']), ('tissue', ['fov-1-scan-1'])])
def test_compute_qc_metrics(gaussian_blur, bin_file_folder, fovs):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define a sample panel, leave panel correctness/incorrectness test for mibi_bin_tools
        panel = pd.DataFrame([{
            'Mass': 89,
            'Target': 'SMA',
            'Start': 88.7,
            'Stop': 89.0,
        }])

        # write the panel to csv
        panel_path = os.path.join(temp_dir, 'sample_panel.csv')
        panel.to_csv(panel_path, index=False)

        # define the full path to the bin file folder
        bin_file_path = os.path.join(Path(__file__).parent, 'data', bin_file_folder)

        # define a sample qc_path to write to
        qc_path = os.path.join(temp_dir, 'sample_qc_dir')

        # bin file error check
        with pytest.raises(FileNotFoundError):
            qc_comp.compute_qc_metrics(
                'bad_bin_path', fovs[0], panel_path, gaussian_blur
            )

        # panel file error check
        with pytest.raises(FileNotFoundError):
            qc_comp.compute_qc_metrics(
                bin_file_path, fovs[0], 'bad_panel_path', gaussian_blur
            )

        # fov error check
        with pytest.raises(FileNotFoundError):
            qc_comp.compute_qc_metrics(
                bin_file_path, 'bad_fov', panel_path, gaussian_blur
            )

        # first time: create new files, also asserts qc_path is created
        qc_comp.compute_qc_metrics(
            bin_file_path, fovs[0], panel_path, gaussian_blur
        )

        nonzero_mean_path = os.path.join(
            bin_file_path, '%s_nonzero_mean_stats.csv' % fovs[0]
        )
        total_intensity_path = os.path.join(
            bin_file_path, '%s_total_intensity_stats.csv' % fovs[0]
        )
        percentile_99_9_path = os.path.join(
            bin_file_path, '%s_percentile_99_9_stats.csv' % fovs[0]
        )

        # assert the QC files for fov-1-scan-1 were created
        assert os.path.exists(nonzero_mean_path)
        assert os.path.exists(total_intensity_path)
        assert os.path.exists(percentile_99_9_path)

        # read in the QC data
        nm = pd.read_csv(nonzero_mean_path)
        ti = pd.read_csv(total_intensity_path)
        p99_9 = pd.read_csv(percentile_99_9_path)

        # assert the column names are correct
        assert list(nm.columns.values) == ['fov', 'channel', 'Non-zero mean intensity']
        assert list(ti.columns.values) == ['fov', 'channel', 'Total intensity']
        assert list(p99_9.columns.values) == ['fov', 'channel', '99.9 intensity value']

        # assert the correct FOV was written
        assert list(nm['fov']) == [fovs[0]]
        assert list(nm['fov']) == list(ti['fov']) == list(p99_9['fov'])

        # assert the correct channels were written
        assert list(nm['channel']) == ['SMA']
        assert list(nm['channel']) == list(ti['channel']) == list(p99_9['channel'])


@parametrize('fovs', [['fov-1-scan-1'], ['fov-1-scan-1', 'fov-2-scan-1']])
def test_combine_qc_metrics(fovs):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define a dummy list of channels
        chans = ['SMA', 'Vimentin', 'Au']

        # define a sample bin_file_path
        bin_file_path = os.path.join(temp_dir, 'sample_qc_dir')
        os.mkdir(bin_file_path)

        # put some random stuff in bin_file_path, test that this does not affect aggregation
        pd.DataFrame().to_csv(os.path.join(bin_file_path, 'random.csv'))
        Path(os.path.join(bin_file_path, 'random.txt')).touch()

        # add existing combined .csv files, this also should not affect aggregation
        pd.DataFrame().to_csv(os.path.join(bin_file_path, 'combined_nonzero_mean_stats.csv'))
        pd.DataFrame().to_csv(os.path.join(bin_file_path, 'combined_total_intensity_stats.csv'))
        pd.DataFrame().to_csv(os.path.join(bin_file_path, 'combined_percentile_99_9_stats.csv'))

        # define sample dataframes for each fov to write to qc_path
        for fov in fovs:
            fov_nm_data = pd.DataFrame(
                np.random.rand(3, 3), columns=['fov', 'channel', 'Non-zero mean intensity']
            )

            fov_ti_data = pd.DataFrame(
                np.random.rand(3, 3), columns=['fov', 'channel', 'Total intensity']
            )

            fov_99_9_data = pd.DataFrame(
                np.random.rand(3, 3), columns=['fov', 'channel', '99.9 intensity value']
            )

            # write the FOV names in
            fov_nm_data['fov'] = fov
            fov_ti_data['fov'] = fov
            fov_99_9_data['fov'] = fov

            # write the channel names in
            fov_nm_data['channel'] = chans
            fov_ti_data['channel'] = chans
            fov_99_9_data['channel'] = chans

            fov_nm_data.to_csv(
                os.path.join(bin_file_path, '%s_nonzero_mean_stats.csv') % fov, index=False
            )
            fov_ti_data.to_csv(
                os.path.join(bin_file_path, '%s_total_intensity_stats.csv') % fov, index=False
            )
            fov_99_9_data.to_csv(
                os.path.join(bin_file_path, '%s_percentile_99_9_stats.csv') % fov, index=False
            )

        # run the aggregation function
        qc_comp.combine_qc_metrics(bin_file_path)

        # assert the correct file names were written
        combined_nm_path = os.path.join(bin_file_path, 'combined_nonzero_mean_stats.csv')
        combined_ti_path = os.path.join(bin_file_path, 'combined_total_intensity_stats.csv')
        combined_p99_9_path = os.path.join(bin_file_path, 'combined_percentile_99_9_stats.csv')

        assert os.path.exists(combined_nm_path)
        assert os.path.exists(combined_ti_path)
        assert os.path.exists(combined_p99_9_path)

        # read in the QC data
        nm = pd.read_csv(combined_nm_path)
        ti = pd.read_csv(combined_ti_path)
        p99_9 = pd.read_csv(combined_p99_9_path)

        # assert the column names are correct
        assert list(nm.columns.values) == ['fov', 'channel', 'Non-zero mean intensity']
        assert list(ti.columns.values) == ['fov', 'channel', 'Total intensity']
        assert list(p99_9.columns.values) == ['fov', 'channel', '99.9 intensity value']

        # assert the correct FOVs are written
        assert list(nm['fov']) == list(np.repeat(fovs, len(chans)))
        assert list(nm['fov']) == list(ti['fov']) == list(p99_9['fov'])

        # assert the correct channels are written
        assert list(nm['channel']) == chans * len(fovs)
        assert list(nm['channel']) == list(ti['channel']) == list(p99_9['channel'])


def test_visualize_qc_metrics():
    # define the channels to use
    chans = ['chan0', 'chan1', 'chan2']

    # define the fov names to use for each channel
    fov_batches = [['fov0', 'fov1'], ['fov2', 'fov3'], ['fov4', 'fov5']]

    # define the test melted DataFrame for an arbitrary QC metric
    sample_qc_metric_data = pd.DataFrame()

    # for each channel append a random set of data for each fov associated with the QC metric
    for chan, fovs in zip(chans, fov_batches):
        chan_data = pd.DataFrame(np.random.rand(len(fovs)), columns=['sample_qc_metric'])
        chan_data['fov'] = fovs
        chan_data['channel'] = chan

        sample_qc_metric_data = pd.concat([sample_qc_metric_data, chan_data])

    with tempfile.TemporaryDirectory() as temp_dir:
        # test without saving
        qc_comp.visualize_qc_metrics(sample_qc_metric_data, 'sample_qc_metric')
        assert not os.path.exists(os.path.join(temp_dir, 'sample_qc_metric_barplot_stats.png'))

        # test with saving
        qc_comp.visualize_qc_metrics(sample_qc_metric_data, 'sample_qc_metric', save_dir=temp_dir)
        assert os.path.exists(os.path.join(temp_dir, 'sample_qc_metric_barplot_stats.png'))
