import os
import shutil
import tempfile
import time
import warnings
from multiprocessing.pool import ThreadPool as Pool
from pathlib import Path
from unittest.mock import patch

import pytest
from pytest_cases import parametrize_with_cases
from tmi import io_utils

from toffy.fov_watcher import start_watcher
from toffy.json_utils import write_json_file
from toffy.test_utils import (RunStructureCases, RunStructureTestContext,
                              WatcherCases, mock_visualize_mph,
                              mock_visualize_qc_metrics)
from toffy.watcher_callbacks import build_callbacks

COMBINED_DATA_PATH = os.path.join(Path(__file__).parent, 'data', 'combined')
RUN_DIR_NAME = 'run_XXX'

SLOW_COPY_INTERVAL_S = 1


def _slow_copy_sample_tissue_data(dest: str, delta: int = 10, one_blank: bool = False):
    """slowly copies files from ./data/tissue/

    Args:
        dest (str):
            Where to copy tissue files to
        delta (int):
            Time (in seconds) between each file copy
    """

    for tissue_file in sorted(os.listdir(COMBINED_DATA_PATH)):
        time.sleep(delta)
        if one_blank and '.bin' in tissue_file:
            # create blank (0 size) file
            open(os.path.join(dest, tissue_file), 'w').close()
            one_blank = False
        else:
            shutil.copy(os.path.join(COMBINED_DATA_PATH, tissue_file), dest)


COMBINED_RUN_JSON_SPOOF = {
    'fovs': [
        {'runOrder': 1, 'scanCount': 2, 'frameSizePixels': {'width': 32, 'height': 32}},
        {'runOrder': 2, 'scanCount': 1, 'frameSizePixels': {'width': 32, 'height': 32}},
        {'runOrder': 3, 'scanCount': 1, 'frameSizePixels': {'width': 32, 'height': 32},
         'standardTarget': "Molybdenum Foil"}
    ],
}


@parametrize_with_cases('run_json, expected_files', cases=RunStructureCases)
def test_run_structure(run_json, expected_files):
    with RunStructureTestContext(run_json, files=expected_files) as (tmpdir, run_structure):
        for file in expected_files:
            run_structure.check_run_condition(os.path.join(tmpdir, file))
        assert all(run_structure.check_fov_progress().values())

        # check for hidden files
        with pytest.warns(Warning, match="is not a valid FOV file and will be skipped"):
            exist, name = run_structure.check_run_condition(os.path.join(tmpdir, '.fake_file.txt'))
        assert not exist and name == ''

        # check for fake files
        with pytest.warns(Warning, match="This should be unreachable..."):
            exist, name = run_structure.check_run_condition(os.path.join(tmpdir, 'fake_file.txt'))
        assert not exist and name == ''


@patch('toffy.watcher_callbacks.visualize_qc_metrics', side_effect=mock_visualize_qc_metrics)
@patch('toffy.watcher_callbacks.visualize_mph', side_effect=mock_visualize_mph)
@pytest.mark.parametrize('add_blank', [False, True])
@parametrize_with_cases('run_cbs, fov_cbs, kwargs, validators', cases=WatcherCases)
def test_watcher(mock_viz_qc, mock_viz_mph, run_cbs, inter_cbs, fov_cbs,
                 kwargs, validators, add_blank):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:

            tiff_out_dir = os.path.join(tmpdir, 'cb_0', RUN_DIR_NAME)
            qc_out_dir = os.path.join(tmpdir, 'cb_1', RUN_DIR_NAME)
            mph_out_dir = os.path.join(tmpdir, 'cb_2', RUN_DIR_NAME)
            plot_dir = os.path.join(tmpdir, 'cb_2_plots', RUN_DIR_NAME)
            pulse_out_dir = os.path.join(tmpdir, 'cb_3', RUN_DIR_NAME)
            stitched_dir = os.path.join(tmpdir, 'cb_0', RUN_DIR_NAME, 'stitched_images')

            # add directories to kwargs
            kwargs['tiff_out_dir'] = tiff_out_dir
            kwargs['qc_out_dir'] = qc_out_dir
            kwargs['mph_out_dir'] = mph_out_dir
            kwargs['pulse_out_dir'] = pulse_out_dir
            kwargs['plot_dir'] = plot_dir

            run_data = os.path.join(tmpdir, 'test_run')
            log_out = os.path.join(tmpdir, 'log_output')
            os.makedirs(run_data)

            fov_callback, run_callback, inter_callback = build_callbacks(
                run_cbs, inter_cbs, fov_cbs, **kwargs
            )
            write_json_file(json_path=os.path.join(run_data, 'test_run.json'),
                            json_object=COMBINED_RUN_JSON_SPOOF)

            # `_slow_copy_sample_tissue_data` mimics the instrument computer uploading data to the
            # client access computer.  `start_watcher` is made async here since these processes
            # wouldn't block each other in normal use

            with Pool(processes=4) as pool:
                pool.apply_async(
                    _slow_copy_sample_tissue_data,
                    (run_data, SLOW_COPY_INTERVAL_S, add_blank)
                )

                # watcher completion is checked every second
                # zero-size files are halted for 1 second or until they have non zero-size
                res_scan = pool.apply_async(
                    start_watcher,
                    (run_data, log_out, fov_callback, run_callback, inter_callback,
                     1, SLOW_COPY_INTERVAL_S)
                )

                res_scan.get()

            with open(os.path.join(log_out, 'test_run_log.txt')) as f:
                logtxt = f.read()
                assert add_blank == ("non-zero file size..." in logtxt)

            fovs = [
                bin_file.split('.')[0]
                for bin_file in sorted(io_utils.list_files(COMBINED_DATA_PATH, substrs=['.bin']))
            ]

            # callbacks are not performed on moly points, remove fov-3-scan-1
            bad_fovs = [fovs[-1]]
            fovs = fovs[:-1]

            # callbacks are not performed for skipped fovs, remove blank fov
            if add_blank:
                bad_fovs.append(fovs[0])
                fovs = fovs[1:]

            # extract tiffs check
            validators[0](os.path.join(tmpdir, 'cb_0', RUN_DIR_NAME), fovs, bad_fovs)

            # qc check
            validators[1](os.path.join(tmpdir, 'cb_1', RUN_DIR_NAME), fovs, bad_fovs)

            # mph check
            validators[2](os.path.join(tmpdir, 'cb_2', RUN_DIR_NAME),
                          os.path.join(tmpdir, 'cb_2_plots', RUN_DIR_NAME), fovs, bad_fovs)

            # stitch images check
            validators[3](os.path.join(tmpdir, 'cb_0', RUN_DIR_NAME, 'stitched_images'))

            # pulse heights check
            validators[4](os.path.join(tmpdir, 'cb_3', RUN_DIR_NAME), fovs, bad_fovs)

    except OSError:
        warnings.warn('Temporary file cleanup was incomplete.')
