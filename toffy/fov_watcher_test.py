import os
import shutil
import time
import tempfile
import json
from pathlib import Path
from multiprocessing.pool import ThreadPool as Pool

import pytest
from pytest_cases import parametrize_with_cases

from mibi_bin_tools import io_utils

from toffy.test_utils import WatcherCases, RunStructureTestContext, RunStructureCases
from toffy.fov_watcher import start_watcher
from toffy.watcher_callbacks import build_callbacks
from toffy.json_utils import write_json_file

TISSUE_DATA_PATH = os.path.join(Path(__file__).parent, 'data', 'tissue')
RUN_DIR_NAME = 'run_XXX'


def _slow_copy_sample_tissue_data(dest: str, delta: int = 10, one_blank: bool = False):
    """slowly copies files from ./data/tissue/

    Args:
        dest (str):
            Where to copy tissue files to
        delta (int):
            Time (in seconds) between each file copy
    """

    for tissue_file in sorted(os.listdir(TISSUE_DATA_PATH)):
        time.sleep(delta)
        if one_blank and '.bin' in tissue_file:
            # create blank (0 size) file
            open(os.path.join(dest, tissue_file), 'w').close()
            one_blank = False
        else:
            shutil.copy(os.path.join(TISSUE_DATA_PATH, tissue_file), dest)


TISSUE_RUN_JSON_SPOOF = {
    'fovs': [
        {'runOrder': 1, 'scanCount': 1},
        {'runOrder': 2, 'scanCount': 1},
    ],
}


@parametrize_with_cases('run_json, expected_files', cases=RunStructureCases)
def test_run_structure(run_json, expected_files):
    with RunStructureTestContext(run_json, files=expected_files) as (tmpdir, run_structure):
        for file in expected_files:
            run_structure.check_run_condition(os.path.join(tmpdir, file))
        assert(all(run_structure.check_fov_progress().values()))

        with pytest.raises(FileNotFoundError):
            run_structure.check_run_condition(os.path.join(tmpdir, 'fake_file.txt'))


# TODO: add tests for per_run when per_run callbacks are created
@pytest.mark.parametrize('add_blank', [False, True])
@parametrize_with_cases('run_cbs, fov_cbs, kwargs, validators', cases=WatcherCases)
def test_watcher(run_cbs, fov_cbs, kwargs, validators, add_blank):
    with tempfile.TemporaryDirectory() as tmpdir:

        tiff_out_dir = os.path.join(tmpdir, 'cb_0', RUN_DIR_NAME)
        qc_out_dir = os.path.join(tmpdir, 'cb_1', RUN_DIR_NAME)

        # add directories to kwargs
        kwargs['tiff_out_dir'] = tiff_out_dir
        kwargs['qc_out_dir'] = qc_out_dir

        run_data = os.path.join(tmpdir, 'test_run')
        log_out = os.path.join(tmpdir, 'log_output')
        os.makedirs(run_data)

        fov_callback, run_callback = build_callbacks(run_cbs, fov_cbs, **kwargs)

        write_json_file(json_path=os.path.join(run_data, 'test_run.json'), 
            json_object=TISSUE_RUN_JSON_SPOOF)

        # `_slow_copy_sample_tissue_data` mimics the instrument computer uploading data to the
        # client access computer.  `start_watcher` is made async here since these processes
        # wouldn't block each other in normal use
        with Pool(processes=4) as pool:
            pool.apply_async(_slow_copy_sample_tissue_data, (run_data, 6, add_blank))

            # watcher completion is checked every 2 seconds
            # zero-size files are halted for 6 seconds or until they have non zero-size
            res_scan = pool.apply_async(
                start_watcher,
                (run_data, log_out, fov_callback, run_callback, 2, 6)
            )

            res_scan.get()

        with open(os.path.join(log_out, 'test_run_log.txt')) as f:
            logtxt = f.read()
            assert(add_blank == ("non-zero file size..." in logtxt))

        fovs = [
            bin_file.split('.')[0]
            for bin_file in sorted(io_utils.list_files(TISSUE_DATA_PATH, substrs=['.bin']))
        ]

        # callbacks are not performed for skipped fovs
        if add_blank:
            fovs = fovs[1:]

        for i, validator in enumerate(validators):
            validator(os.path.join(tmpdir, f'cb_{i}', RUN_DIR_NAME), fovs)

    pass
