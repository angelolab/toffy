import os
import shutil
import time
import tempfile
import json
from pathlib import Path
from multiprocessing.pool import ThreadPool as Pool

from pytest_cases import parametrize_with_cases

from toffy.test_utils import WatcherCases
from toffy.fov_watcher import start_watcher

TISSUE_DATA_PATH = os.path.join(Path(__file__).parent, 'data', 'tissue')


def _slow_copy_sample_tissue_data(dest: str, delta: int = 10):
    """slowly copies files from ./data/tissue/

    Args:
        dest (str):
            Where to copy tissue files to
        delta (int):
            Time (in seconds) between each file copy
    """

    print('copytime')

    for tissue_file in os.listdir(TISSUE_DATA_PATH):
        shutil.copy(os.path.join(TISSUE_DATA_PATH, tissue_file), dest)
        time.sleep(delta)
    print('copies done!')


TISSUE_RUN_JSON_SPOOF = {
    'fovs': [
        {'runOrder': 1, 'scanCount': 1},
        {'runOrder': 2, 'scanCount': 1},
    ],
}


@parametrize_with_cases('per_fov, per_run', cases=WatcherCases)
def test_watcher(per_fov, per_run):
    with tempfile.TemporaryDirectory() as tmpdir:

        run_data = os.path.join(tmpdir, 'test_run')
        watch_out = os.path.join(tmpdir, 'watcher_output')
        os.makedirs(run_data)
        os.makedirs(watch_out)

        with open(os.path.join(run_data, 'test_run.json'), 'w') as f:
            json.dump(TISSUE_RUN_JSON_SPOOF, f)

        with Pool(processes=4) as pool:
            pool.apply_async(_slow_copy_sample_tissue_data, (run_data, 6))
            res_scan = pool.apply_async(start_watcher, (run_data, watch_out, per_fov, per_run, 2))

            res_scan.get()

        outs = os.listdir(os.path.join(watch_out, 'test_run'))

        print(outs)
        print(os.listdir(os.path.join(watch_out, 'test_run', 'extract')))

        with open(os.path.join(watch_out, 'test_run', 'log.txt')) as f:
            print(f.read())

    pass
