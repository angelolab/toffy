import os
import tempfile

import numpy as np
import pytest
from alpineer import io_utils

from toffy import reorg
from toffy.json_utils import write_json_file

from .utils import test_utils


def create_sample_fov_dirs(fovs, base_dir):
    # make fov subdirectories
    for fov in fovs:
        os.mkdir(os.path.join(base_dir, fov))


def remove_fov_dirs(base_dir):
    # delete fov subdirectories
    fovs = io_utils.list_folders(base_dir)
    for fov in fovs:
        os.rmdir(os.path.join(base_dir, fov))


def test_merge_partial_runs(tmpdir):
    run_names = ["run1_start", "run1_finish", "run2", "run3", "run3_restart"]
    fov_names = [
        ["fov1"],
        ["fov2", "fov3", "fov4"],
        ["fov1"],
        ["fov1", "fov2"],
        ["fov3", "fov5"],
    ]

    for i in range(len(run_names)):
        run_path = os.path.join(tmpdir, run_names[i])
        os.makedirs(run_path)
        fovs = fov_names[i]
        create_sample_fov_dirs(fovs=fovs, base_dir=run_path)

    # merge runs
    reorg.merge_partial_runs(cohort_dir=tmpdir, run_string="run1")
    reorg.merge_partial_runs(cohort_dir=tmpdir, run_string="run3")

    # correct runs have been combined
    merged_run_folders = io_utils.list_folders(tmpdir)
    merged_run_folders.sort()
    assert np.array_equal(merged_run_folders, ["run1", "run2", "run3"])

    # correct sub-folders within each run
    merged_fov_folders_run1 = io_utils.list_folders(os.path.join(tmpdir, "run1"))
    merged_fov_folders_run1.sort()
    assert np.array_equal(merged_fov_folders_run1, ["fov1", "fov2", "fov3", "fov4"])

    merged_fov_folders_run3 = io_utils.list_folders(os.path.join(tmpdir, "run3"))
    merged_fov_folders_run3.sort()
    assert np.array_equal(merged_fov_folders_run3, ["fov1", "fov2", "fov3", "fov5"])

    # check that bad string raises error
    with pytest.raises(ValueError, match="No matching"):
        reorg.merge_partial_runs(cohort_dir=tmpdir, run_string="run4")

    # check that duplicate FOVs raises error
    for run, fovs in zip(["run_dup", "run_dup2"], [["fov1"], ["fov1", "fov2"]]):
        for fov in fovs:
            os.makedirs(os.path.join(tmpdir, run, fov))
    with pytest.raises(ValueError, match="fov1"):
        reorg.merge_partial_runs(cohort_dir=tmpdir, run_string="dup")

    # check multiple duplicate FOVs (in different runs) works
    os.makedirs(os.path.join(tmpdir, "run_dup3", "fov2"))
    os.makedirs(os.path.join(tmpdir, "run_dup4", "fov2"))
    with pytest.raises(ValueError, match="fov2"):
        reorg.merge_partial_runs(cohort_dir=tmpdir, run_string="dup")


def test_combine_runs(tmpdir):
    run_names = ["run1", "run2", "run4"]
    fov_names = [["fov1", "fov2"], ["fov1", "fov3"], ["fov5"]]
    expected_names = []

    # create each fov within each run
    for i in range(len(run_names)):
        run = run_names[i]
        run_path = os.path.join(tmpdir, run)
        os.makedirs(run_path)

        fovs = fov_names[i]
        for fov in fovs:
            os.makedirs(os.path.join(run_path, fov))
            expected_names.append(run + "_" + fov)

    reorg.combine_runs(tmpdir)
    created_names = io_utils.list_folders(os.path.join(tmpdir, "image_data"))
    created_names.sort()

    assert np.array_equal(created_names, expected_names)


def test_rename_fov_dirs():
    with tempfile.TemporaryDirectory() as base_dir:
        # create run file and fov folder directories
        dirs = ["run_folder", "fov_folder"]
        for directory in dirs:
            os.mkdir(os.path.join(base_dir, directory))
        run_dir = os.path.join(base_dir, "run_folder")
        fov_dir = os.path.join(base_dir, "fov_folder")

        # create existing new directory
        not_new_dir = os.path.join(base_dir, "not_new_directory")
        os.mkdir(not_new_dir)

        # existing directory for new_dir should raise an error
        with pytest.raises(ValueError, match="already exists"):
            reorg.rename_fov_dirs(run_dir, fov_dir, not_new_dir)

        # regular sample run data
        ex_name = ["custom_1", "custom_2", "custom_3"]
        ex_run_order = list(range(1, 4))
        ex_scan_count = [1, 2, 1]

        # create a json path with the sample data
        ex_run_path = test_utils.create_sample_run(ex_name, ex_run_order, ex_scan_count, True)

        # bad sample run data
        bad_run = test_utils.create_sample_run(ex_name, ex_run_order, ex_scan_count, True, bad=True)

        # bad run file data should raise an error
        with pytest.raises(KeyError):
            reorg.rename_fov_dirs(bad_run, fov_dir)

        # create already renamed fov folders
        renamed_fovs = ["custom_1", "custom_2-1", "custom_2-2", "custom_3"]
        create_sample_fov_dirs(renamed_fovs, fov_dir)

        # fov folders already renamed should raise an error
        with pytest.raises(ValueError, match=r"already been renamed"):
            reorg.rename_fov_dirs(ex_run_path, fov_dir)
        remove_fov_dirs(fov_dir)

        # create correct fov folders
        correct_fovs = ["fov-1-scan-1", "fov-2-scan-1", "fov-2-scan-2", "fov-3-scan-1"]
        create_sample_fov_dirs(correct_fovs, fov_dir)

        # test successful renaming to new dir
        reorg.rename_fov_dirs(ex_run_path, fov_dir, os.path.join(base_dir, "new_directory"))
        new_dir = os.path.join(base_dir, "new_directory")
        assert set(io_utils.list_folders(new_dir)) == set(renamed_fovs)
        remove_fov_dirs(new_dir)

        # test successful renaming
        reorg.rename_fov_dirs(ex_run_path, fov_dir)
        assert set(io_utils.list_folders(fov_dir)) == set(renamed_fovs)
        remove_fov_dirs(fov_dir)

        # create not enough fov folders
        less_fovs = ["fov-1-scan-1", "fov-2-scan-1", "fov-3-scan-1"]
        create_sample_fov_dirs(less_fovs, fov_dir)

        # fovs in rule file without an existing dir should raise an error
        with pytest.warns(UserWarning, match="Not all values"):
            reorg.rename_fov_dirs(ex_run_path, fov_dir)
        remove_fov_dirs(fov_dir)

        # create extra fov folders
        extra_fovs = [
            "fov-1-scan-1",
            "fov-2-scan-1",
            "fov-2-scan-2",
            "fov-3-scan-1",
            "fov-3-scan-3",
        ]
        create_sample_fov_dirs(extra_fovs, fov_dir)

        # extra dirs not listed in run file should raise an error
        with pytest.raises(ValueError, match="Not all values"):
            reorg.rename_fov_dirs(ex_run_path, fov_dir)
        remove_fov_dirs(fov_dir)


def test_rename_fovs_in_cohort(tmpdir):
    bin_base_dir = os.path.join(tmpdir, "bins")
    cohort_dir = os.path.join(tmpdir, "cohort")
    processed_base_dir = os.path.join(tmpdir, "processed")

    # create directories
    for dir in [bin_base_dir, cohort_dir, processed_base_dir]:
        os.makedirs(dir)

    # create json for first run
    custom_names1 = ["custom_1", "custom_2", "custom_3"]
    run_order = list(range(1, 4))
    scan_count = [1, 1, 1]
    json1 = test_utils.create_sample_run(custom_names1, run_order, scan_count)

    # create json for second run
    custom_names2 = ["custom_5", "custom_6", "custom_7"]
    json2 = test_utils.create_sample_run(custom_names2, run_order, scan_count)

    # create json in run-specific subfolder
    runs = ["run1", "run2"]
    for run, json_file in zip(runs, [json1, json2]):
        run_path = os.path.join(bin_base_dir, run)
        os.makedirs(run_path)
        json_path = os.path.join(run_path, run + ".json")

        write_json_file(json_path=json_path, json_object=json_file)

    # create run FOVs
    fovs = ["fov-1-scan-1", "fov-2-scan-1", "fov-3-scan-1"]
    for run in runs:
        run_path = os.path.join(processed_base_dir, run)
        os.makedirs(run_path)

        create_sample_fov_dirs(fovs=fovs, base_dir=run_path)

    reorg.rename_fovs_in_cohort(
        run_names=runs,
        processed_base_dir=processed_base_dir,
        cohort_path=cohort_dir,
        bin_base_dir=bin_base_dir,
    )

    for run, fov_names in zip(runs, [custom_names1, custom_names2]):
        renamed_dir = os.path.join(cohort_dir, run)
        renamed_fovs = io_utils.list_folders(renamed_dir)

        assert set(renamed_fovs) == set(fov_names)
