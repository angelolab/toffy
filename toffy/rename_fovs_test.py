import tempfile
import json
import os
import pytest

# from ark.utils import test_utils
from toffy import rename_fovs


def create_sample_run(name_list, run_order_list, scan_count_list, create_json=False):
    fov_list = []
    sample_run = {"fovs": fov_list}

    for name, run_order, scan_count in zip(name_list, run_order_list, scan_count_list):
        ex_fov = {
            "scanCount": scan_count,
            "runOrder": run_order,
            "name": name
        }
        fov_list.append(ex_fov)

    for fov in sample_run.get('fovs', ()):
        if fov.get('name') is None:
            del fov['name']

    if create_json:
        temp = tempfile.NamedTemporaryFile(mode="w")
        json.dump(sample_run, temp)
        print(temp.name)
        return temp.name

    return sample_run


def test_check_unnamed_fovs():
    ex_name = ['MoQC', None, 'tonsil_bottom', 'moly_qc_tissue', None]
    ex_run_order = list(range(1, 6))
    ex_scan_count = list(range(1, 6))
    ex_run = create_sample_run(ex_name, ex_run_order, ex_scan_count)

    rename_fovs.check_unnamed_fovs(ex_run)
    for fov in ex_run.get('fovs', ()):
        assert fov.get('name') is not None


def test_rename_fov_dirs():
    with tempfile.TemporaryDirectory() as base_dir:
        # create run file and fov folder directories
        dirs = ['run_folder', 'fov_folder']
        for directory in dirs:
            os.mkdir(os.path.join(base_dir, directory))
        run_dir = os.path.join(base_dir, 'run_folder')
        fov_dir = os.path.join(base_dir, 'fov_folder')

        # create existing new directory
        os.mkdir(os.path.join(base_dir, 'new_directory'))
        not_new_dir = os.path.join(base_dir, 'new_directory')

        # test existing directory for new_dir
        with pytest.raises(ValueError):
            rename_fovs.rename_fov_dirs(run_dir, fov_dir, not_new_dir)



# test_rename_fov_dirs()
