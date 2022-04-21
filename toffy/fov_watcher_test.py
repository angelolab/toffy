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

    print('copytime')

    for tissue_file in sorted(os.listdir(TISSUE_DATA_PATH)):
        time.sleep(delta)
        if one_blank and '.bin' in tissue_file:
            # create blank (0 size) file
            open(os.path.join(dest, tissue_file), 'w').close()
            one_blank = False
        else:
            shutil.copy(os.path.join(TISSUE_DATA_PATH, tissue_file), dest)
    print('copies done!')


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
@parametrize_with_cases('per_fov_partial, per_run, validators', cases=WatcherCases)
def test_watcher(per_fov_partial, per_run, validators, add_blank):
    with tempfile.TemporaryDirectory() as tmpdir:
        per_fov = []
        for i, func in enumerate(per_fov_partial):
            cb_dir = os.path.join(tmpdir, f'cb_{i}', RUN_DIR_NAME)
            os.makedirs(cb_dir)
            per_fov.append(func(cb_dir))

        run_data = os.path.join(tmpdir, 'test_run')
        log_out = os.path.join(tmpdir, 'log_output')
        os.makedirs(run_data)
        os.makedirs(log_out)

        with open(os.path.join(run_data, 'test_run.json'), 'w') as f:
            json.dump(TISSUE_RUN_JSON_SPOOF, f)

        with Pool(processes=4) as pool:
            pool.apply_async(_slow_copy_sample_tissue_data, (run_data, 6, add_blank))
            res_scan = pool.apply_async(start_watcher, (run_data, log_out, per_fov, per_run, 2, 6))

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
