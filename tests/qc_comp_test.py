import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from alpineer import io_utils, load_utils, misc_utils
from mibi_bin_tools import bin_files

from toffy import qc_comp, settings
from toffy.mibitracker_utils import MibiRequests

parametrize = pytest.mark.parametrize


def test_compute_nonzero_mean_intensity():
    # test on a zero array
    sample_img_arr = np.zeros((3, 3))
    sample_nonzero_mean = qc_comp.compute_nonzero_mean_intensity(sample_img_arr)
    assert sample_nonzero_mean == 0

    # test on a non-zero array
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


def test_sort_bin_file_fovs():
    # test without suffix ignore
    fov_list = [
        "fov-2-scan-2",
        "fov-10-scan-1",
        "fov-5-scan-3",
        "fov-2-scan-10",
        "fov-200-scan-4",
    ]
    fov_list_sorted = qc_comp.sort_bin_file_fovs(fov_list)
    assert fov_list_sorted == [
        "fov-2-scan-2",
        "fov-2-scan-10",
        "fov-5-scan-3",
        "fov-10-scan-1",
        "fov-200-scan-4",
    ]

    # test with a suffix on some fovs
    fov_list_some_suffix = fov_list[:]
    fov_list_some_suffix[:2] = [f + "_suffix.csv" for f in fov_list[:2]]
    fov_list_sorted = qc_comp.sort_bin_file_fovs(fov_list_some_suffix, suffix_ignore="_suffix.csv")
    assert fov_list_sorted == [
        "fov-2-scan-2_suffix.csv",
        "fov-2-scan-10",
        "fov-5-scan-3",
        "fov-10-scan-1_suffix.csv",
        "fov-200-scan-4",
    ]

    # test with a suffix on all fovs
    fov_list_all_suffix = [f + "_suffix.csv" for f in fov_list]
    fov_list_sorted = qc_comp.sort_bin_file_fovs(fov_list_all_suffix, suffix_ignore="_suffix.csv")
    assert fov_list_sorted == [
        "fov-2-scan-2_suffix.csv",
        "fov-2-scan-10_suffix.csv",
        "fov-5-scan-3_suffix.csv",
        "fov-10-scan-1_suffix.csv",
        "fov-200-scan-4_suffix.csv",
    ]


# NOTE: we don't need to test iteration over multiple FOVs because
# test_compute_qc_metrics computes on 1 FOV at a time, fov-3-scan-1 is a moly point
@parametrize("gaussian_blur", [False, True])
@parametrize(
    "bin_file_folder, fovs",
    [("combined", ["fov-3-scan-1"]), ("combined", ["fov-1-scan-1"])],
)
def test_compute_qc_metrics(gaussian_blur, bin_file_folder, fovs):
    with tempfile.TemporaryDirectory() as temp_dir:
        # define a sample panel, leave panel correctness/incorrectness test for mibi_bin_tools
        panel = pd.DataFrame(
            [
                {
                    "Mass": 89,
                    "Target": "SMA",
                    "Start": 88.7,
                    "Stop": 89.0,
                }
            ]
        )

        # define the full path to the bin file folder
        bin_file_path = os.path.join(Path(__file__).parents[1], "data", bin_file_folder)

        # define a sample extraction directory
        extracted_imgs_path = os.path.join(temp_dir, "extracted_images", bin_file_folder)
        os.makedirs(extracted_imgs_path)
        bin_files.extract_bin_files(bin_file_path, extracted_imgs_path, panel=panel)

        # define a sample qc_path to write to
        qc_path = os.path.join(temp_dir, "sample_qc_dir")

        # extraction folder error check
        with pytest.raises(FileNotFoundError):
            qc_comp.compute_qc_metrics("bad_extraction_path", fovs[0], gaussian_blur)

        # fov error check
        with pytest.raises(FileNotFoundError):
            qc_comp.compute_qc_metrics(extracted_imgs_path, "bad_fov", gaussian_blur)

        # first time: create new files, also asserts qc_path is created
        qc_comp.compute_qc_metrics(
            extracted_imgs_path, fovs[0], gaussian_blur, save_csv=bin_file_path
        )

        for ms, mc in zip(settings.QC_SUFFIXES, settings.QC_COLUMNS):
            # assert the file for this QC metric was created
            metric_path = os.path.join(bin_file_path, "%s_%s.csv" % (fovs[0], ms))
            assert os.path.exists(metric_path)

            # read the data for this QC metric
            metric_data = pd.read_csv(metric_path)

            # assert the column names are correct
            assert list(metric_data.columns.values) == ["fov", "channel", mc]

            # assert the correct FOV was written
            assert list(metric_data["fov"]) == [fovs[0]]

            # assert the correct channels were written
            assert list(metric_data["channel"]) == ["SMA"]


@parametrize("fovs", [["fov-1-scan-1"], ["fov-1-scan-1", "fov-2-scan-1"]])
@parametrize("warn_overwrite_test", [True, False])
def test_combine_qc_metrics(fovs, warn_overwrite_test):
    with tempfile.TemporaryDirectory() as temp_dir:
        # bin folder error check
        with pytest.raises(FileNotFoundError):
            qc_comp.combine_qc_metrics("bad_bin_path")

        # define a dummy list of channels
        chans = ["SMA", "Vimentin", "Au"]

        # define a sample bin_file_path
        bin_file_path = os.path.join(temp_dir, "sample_qc_dir")
        os.mkdir(bin_file_path)

        # put some random stuff in bin_file_path, test that this does not affect aggregation
        pd.DataFrame().to_csv(os.path.join(bin_file_path, "random.csv"))
        Path(os.path.join(bin_file_path, "random.txt")).touch()

        # the start value to generate dummy data from for each QC metric
        metric_start_vals = [2, 3, 4]

        # define sample .csv files for each QC metric
        for ms, mv, mc in zip(settings.QC_SUFFIXES, metric_start_vals, settings.QC_COLUMNS):
            # add existing combined .csv files, these should not be included in aggregation
            pd.DataFrame().to_csv(os.path.join(bin_file_path, "combined_%s.csv" % ms))

            # create a sample dataframe for the QC metric
            for i, fov in enumerate(fovs):
                fov_qc_data = pd.DataFrame(np.zeros((3, 3)), columns=["fov", "channel", mc])

                # write the FOV name in
                fov_qc_data["fov"] = fov

                # write the channel names in
                fov_qc_data["channel"] = chans

                # write the QC metric data in, we'll include different values for each
                fov_qc_data[mc] = np.arange(mv * (i + 1), mv * (i + 1) + 3)

                # write the dummy QC data
                fov_qc_data.to_csv(
                    os.path.join(bin_file_path, "%s_%s.csv" % (fov, ms)), index=False
                )

        # create a dummy file if testing the warn_overwrite flag
        if warn_overwrite:
            Path(os.path.join(bin_file_path, "combined_%s.csv" % settings.QC_SUFFIXES[0])).touch()

        # run the aggregation function, testing that both warn_overwrite states work properly
        if warn_overwrite_test:
            with pytest.warns(
                UserWarning,
                match="Removing previously generated combined %s" % settings.QC_SUFFIXES[0],
            ):
                qc_comp.combine_qc_metrics(bin_file_path, warn_overwrite_test)
        else:
            qc_comp.combine_qc_metrics(bin_file_path, warn_overwrite_test)

        for ms, mv, mc in zip(settings.QC_SUFFIXES, metric_start_vals, settings.QC_COLUMNS):
            # assert the combined QC metric file was created
            combined_qc_path = os.path.join(bin_file_path, "combined_%s.csv" % ms)
            assert os.path.exists(combined_qc_path)

            # read in the combined QC data
            metric_data = pd.read_csv(combined_qc_path)

            # assert the column names are correct
            assert list(metric_data.columns.values) == ["fov", "channel", mc]

            # assert the correct FOVs are written
            assert list(metric_data["fov"]) == list(np.repeat(fovs, len(chans)))

            # assert the correct channels are written
            assert list(metric_data["channel"] == chans * len(fovs))

            # assert the correct QC metric values were written
            qc_metric_vals = []
            for i in range(len(fovs)):
                qc_metric_vals.extend(range(mv * (i + 1), mv * (i + 1) + 3))
            assert list(metric_data[mc]) == qc_metric_vals


def test_visualize_qc_metrics():
    # define the channels to use
    chans = ["chan0", "chan1", "chan2"]

    # define the fov names to use for each channel
    fov_batches = [["fov0", "fov1"], ["fov2", "fov3"], ["fov4", "fov5"]]

    # define the supported metrics to iterate over
    metrics = ["Non-zero mean intensity", "Total intensity", "99.9% intensity value"]

    with tempfile.TemporaryDirectory() as temp_dir:
        # save sample combined .csv files for each metric
        for metric in metrics:
            # define the test melted DataFrame for an arbitrary QC metric
            sample_qc_metric_data = pd.DataFrame()

            for chan, fovs in zip(chans, fov_batches):
                chan_data = pd.DataFrame(np.random.rand(len(fovs)), columns=[metric])

                chan_data["fov"] = fovs
                chan_data["channel"] = chan

                sample_qc_metric_data = pd.concat([sample_qc_metric_data, chan_data])

            # get the file name of the combined QC metric .csv file to use
            qc_metric_index = settings.QC_COLUMNS.index(metric)
            qc_metric_suffix = settings.QC_SUFFIXES[qc_metric_index] + ".csv"

            # save the combined data
            sample_qc_metric_data.to_csv(
                os.path.join(temp_dir, "combined_%s" % qc_metric_suffix), index=False
            )

        # pass an invalid metric
        with pytest.raises(ValueError):
            qc_comp.visualize_qc_metrics(metric_name="bad_metric", qc_metric_dir="", save_dir="")

        # pass an invalid qc_metric_dir
        with pytest.raises(FileNotFoundError):
            qc_comp.visualize_qc_metrics("Non-zero mean intensity", "bad_qc_dir", save_dir="")

        # pass a qc_metric_dir without the combined files
        os.mkdir(os.path.join(temp_dir, "empty_qc_dir"))
        with pytest.raises(FileNotFoundError):
            qc_comp.visualize_qc_metrics(
                "Non-zero mean intensity", os.path.join(temp_dir, "empty_qc_dir"), save_dir=""
            )

        # now test the visualization process for each metric
        for metric in metrics:
            # test without saving (should raise an error)
            with pytest.raises(TypeError):
                qc_comp.visualize_qc_metrics(metric, temp_dir)

            # test with saving
            qc_comp.visualize_qc_metrics(metric, temp_dir, save_dir=temp_dir)
            assert os.path.exists(os.path.join(temp_dir, "%s_barplot_stats.png" % metric))


def test_format_img_data():
    # define a sample panel, leave panel correctness/incorrectness test for mibi_bin_tools
    panel = pd.DataFrame(
        [
            {
                "Mass": 89,
                "Target": "SMA",
                "Start": 88.7,
                "Stop": 89.0,
            }
        ]
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # define the full path to the bin file folder
        bin_file_path = os.path.join(Path(__file__).parents[1], "data", "combined")

        # define a sample extraction directory
        extracted_imgs_path = os.path.join(temp_dir, "extracted_images", "combined")
        os.makedirs(extracted_imgs_path)
        bin_files.extract_bin_files(bin_file_path, extracted_imgs_path, panel=panel)

        # retrieve image array from tiffs
        load_data = load_utils.load_imgs_from_tree(extracted_imgs_path, fovs=["fov-1-scan-1"])

        # retrieve image array from bin files
        extracted_data = bin_files.extract_bin_files(
            bin_file_path, out_dir=None, include_fovs=["fov-1-scan-1"], panel=panel
        )

        # check that they have inherently different structures
        assert not load_data.equals(extracted_data)

        # test for successful reformatting
        load_data = qc_comp.format_img_data(load_data)
        assert load_data.equals(extracted_data)
