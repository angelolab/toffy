import itertools
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from alpineer import io_utils, load_utils, test_utils
from mibi_bin_tools import bin_files

from toffy import qc_comp, settings

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
def test_combine_qc_metrics(fovs):
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

        # run the aggregation function
        qc_comp.combine_qc_metrics(bin_file_path)

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
            sample_QCMetricData = pd.DataFrame()

            for chan, fovs in zip(chans, fov_batches):
                chan_data = pd.DataFrame(np.random.rand(len(fovs)), columns=[metric])

                chan_data["fov"] = fovs
                chan_data["channel"] = chan

                sample_QCMetricData = pd.concat([sample_QCMetricData, chan_data])

            # get the file name of the combined QC metric .csv file to use
            qc_metric_index = settings.QC_COLUMNS.index(metric)
            qc_metric_suffix = settings.QC_SUFFIXES[qc_metric_index] + ".csv"

            # save the combined data
            sample_QCMetricData.to_csv(
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


@dataclass
class QCMetricData:
    """
    Contains misc information for a testing set of QC information such as the tma names, fovs,
    channels, and a DataFrame containing testing data.
    """

    tma_name: str
    fov_count: int
    channel_ignore_count: int
    tma_n_m: np.ndarray
    fovs: list[str]
    channels: list[str]
    regex_search_term: re.Pattern
    qc_df: pd.DataFrame


@pytest.fixture(scope="module")
def qc_tmas(rng: np.random.Generator) -> Generator[QCMetricData, None, None]:
    """
    A Fixture which yields a dataclass used containing the QC dataframe with the three metrics,
    the tma name, fovs, channels and the regex search term for RnCm.

    Args:
        rng (np.random.Generator): The random number generator in `conftest.py`.

    Yields:
        Generator[QCMetricData, None, None]: The dataclass containing testing data.
    """
    fov_count: int = 5
    channel_ignore_count: int = len(settings.QC_CHANNEL_IGNORE)
    channel_count: int = 3
    total_n_m_options: np.ndarray = np.arange(0, 13)
    regex_search_term: re.Pattern = re.compile(r"R\+?(\d+)C\+?(\d+)")
    tma_name: str = "Project_TMA1"

    tma_n_m = rng.choice(a=total_n_m_options, size=(fov_count, 2), replace=False)
    fovs: list[str] = [f"Project_TMA1_R{tma[0]}C{tma[1]}" for tma in tma_n_m]

    channels: list[str] = [f"chan_{i}" for i in range(channel_count)] + settings.QC_CHANNEL_IGNORE

    qc_df: pd.DataFrame = pd.DataFrame(
        data={
            "fov": fovs * (channel_count + channel_ignore_count),
            "channel": channels * fov_count,
            **{
                qc_col: rng.random(size=(fov_count * (channel_count + channel_ignore_count)))
                for qc_col in settings.QC_COLUMNS
            },
            "row": np.tile(A=tma_n_m[:, 0], reps=(channel_count + channel_ignore_count)),
            "column": np.tile(A=tma_n_m[:, 1], reps=(channel_count + channel_ignore_count)),
        }
    )

    qc_data = QCMetricData(
        tma_name=tma_name,
        fov_count=fov_count,
        channel_ignore_count=channel_ignore_count,
        tma_n_m=tma_n_m,
        fovs=fovs,
        channels=channels,
        regex_search_term=regex_search_term,
        qc_df=qc_df,
    )

    yield qc_data


def test__get_r_c(qc_tmas: QCMetricData) -> None:
    result = pd.DataFrame()
    result[["R", "C"]]: pd.DataFrame = qc_tmas.qc_df["fov"].apply(
        lambda row: qc_comp._get_r_c(row, qc_tmas.regex_search_term)
    )

    # Make sure the lengths are the same
    assert len(result) == len(qc_tmas.qc_df)

    # Make sure the set of values are the same for Row and Column
    assert set(result["R"]) == set(qc_tmas.tma_n_m[:, 0])
    assert set(result["C"]) == set(qc_tmas.tma_n_m[:, 1])


def test_qc_tma_metrics(tmp_path: Path, qc_tmas: QCMetricData) -> None:
    qc_tma_metrics_dir: Path = tmp_path / "metrics"
    run_path: Path = tmp_path / "the_run"
    qc_tma_metrics_dir.mkdir(parents=True, exist_ok=True)
    run_path.mkdir(parents=True, exist_ok=True)

    _ = test_utils.create_paired_xarray_fovs(
        base_dir=run_path,
        fov_names=qc_tmas.fovs,
        channel_names=qc_tmas.channels,
        img_shape=(20, 20),
    )

    qc_comp.qc_tma_metrics(
        extracted_imgs_path=run_path, qc_tma_metrics_dir=qc_tma_metrics_dir, tma=qc_tmas.tma_name
    )

    for ms in settings.QC_SUFFIXES:
        all_metric_files: str = io_utils.list_files(qc_tma_metrics_dir, substrs=f"{ms}.csv")

        # Get the combined file
        combined_metric_file: str = next(filter(lambda mf: "combined" in mf, all_metric_files))

        # Filter out combined files
        metric_files: list[str] = list(filter(lambda mf: "combined" not in mf, all_metric_files))

        # Make sure that they all have the `tma_name` as the prefix
        assert all(mf.startswith(qc_tmas.tma_name) for mf in all_metric_files)

        # Combined metric file df
        combined_mf_df = pd.read_csv(qc_tma_metrics_dir / combined_metric_file)

        for mf in metric_files:
            mf_df: pd.DataFrame = pd.read_csv(qc_tma_metrics_dir / mf)
            # Merge the dataframes together, and check that "fov", "channel", and metric val
            # Assert that all elements in the metric csv exist in the combined metric csv
            pd.testing.assert_frame_equal(
                left=combined_mf_df.merge(mf_df).iloc[:, 0:3], right=mf_df
            )


def test__create_r_c_tma_matrix(qc_tmas: QCMetricData) -> None:
    x_size, y_size = np.max(qc_tmas.tma_n_m, axis=0)

    for qc_col in settings.QC_COLUMNS:
        r_c_tma_matrix_df = pd.DataFrame()

        r_c_tma_matrix_df[["rc_matrix"]] = qc_tmas.qc_df.groupby(by="channel", sort=True).apply(
            lambda group: qc_comp._create_r_c_tma_matrix(group, y_size, x_size, qc_col)
        )

        # Assert that the shapes are correct.
        for rc_matrix in r_c_tma_matrix_df["rc_matrix"]:
            assert rc_matrix.shape == (y_size, x_size)


@pytest.fixture(scope="function")
def qc_tma_csvs(qc_tmas: QCMetricData, tmp_path: Path) -> Generator[Path, None, None]:
    """
    A fixture which creates and saves combined metric csv files for each QC metric in the
    qc_tmas: QCMetricData dataclass. Yields the path where the combined QC csvs are saved.

    Args:
        qc_tmas (QCMetricData): QC TMA Dataclass, contains testing data.
        tmp_path (Path): A temporary directory to write files for testing.

    Yields:
        Generator[Path, None, None]: The directory where the combined TMA QCs are saved.
    """
    qc_tma_metrics_dir: Path = tmp_path / "metrics"
    qc_tma_metrics_dir.mkdir(parents=True, exist_ok=True)

    for qc_col, qc_suffix in zip(settings.QC_COLUMNS, settings.QC_SUFFIXES):
        qc_tmas.qc_df[["fov", "channel", qc_col, "row", "column"]].to_csv(
            qc_tma_metrics_dir / f"{qc_tmas.tma_name}_combined_{qc_suffix}.csv", index=False
        )
    yield qc_tma_metrics_dir


@parametrize(
    "qc_metrics, channel_exclude",
    [(settings.QC_COLUMNS[:2], None), ([settings.QC_COLUMNS[0]], ["chan_0"]), (None, None)],
)
def test_qc_tma_metrics_rank(
    qc_tma_csvs: Path,
    qc_tmas: QCMetricData,
    qc_metrics: list[str],
    channel_exclude: list[str] | None,
) -> None:
    cmt_data = qc_comp.qc_tma_metrics_rank(
        qc_tma_metrics_dir=qc_tma_csvs,
        tma=qc_tmas.tma_name,
        qc_metrics=qc_metrics,
        channel_exclude=channel_exclude,
    )

    # Make sure only the specified QC metrics ranks get computed.
    assert set(cmt_data.keys()) == set(qc_metrics)

    # Check that the shape is max(Row) x max(Column)
    assert set([(data.shape) for data in cmt_data.values()]) == set([tuple(qc_tmas.tma_n_m.max(0))])


@pytest.fixture(scope="function")
def cohort_data(tmp_path: Path) -> Generator[tuple[Path, Path], None, None]:
    """
    A fixture for generating cohort fovs, and channels for various tissues.

    Args:
        tmp_path (Path): A temporary directory to write files for testing.

    Yields:
        Generator[tuple[Path, Path], None, None]: Yields two directories, one for the cohort data,
        and another for the cohort metrics.
    """
    cohort_dir: Path = tmp_path / "my_cohort"
    cohort_data_dir: Path = cohort_dir / "images"
    cohort_metrics_dir: Path = cohort_dir / "cohort_metrics"

    for directory in [cohort_dir, cohort_data_dir, cohort_metrics_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    tissues: list[str] = [f"tissue{i}" for i in range(3)]
    fov_names, chan_names = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=5)

    chan_names.extend(settings.QC_SUFFIXES)
    fov_names: list[str] = [
        f"{fov}_{tissue}" for fov, tissue in itertools.product(fov_names, tissues)
    ]

    _, _ = test_utils.create_paired_xarray_fovs(
        base_dir=cohort_data_dir, fov_names=fov_names, channel_names=chan_names, img_shape=(20, 20)
    )

    yield (cohort_data_dir, cohort_metrics_dir)


@parametrize("_tissues", [(["tissue"]), (["tissue0"]), (["tissue0", "tissue1", "tissue2"])])
def test_batch_effect_qc_metrics(cohort_data: tuple[Path, Path], _tissues: list[str]):
    _cohort_data_dir, _cohort_qc_metrics_dir = cohort_data

    with pytest.raises(FileNotFoundError):
        qc_comp.batch_effect_qc_metrics(
            cohort_data_dir="bad_cohort_dir",
            qc_cohort_metrics_dir=_cohort_qc_metrics_dir,
            tissues=_tissues,
        )
    with pytest.raises(FileNotFoundError):
        qc_comp.batch_effect_qc_metrics(
            cohort_data_dir=_cohort_data_dir,
            qc_cohort_metrics_dir="bad_cohort_metric_dr",
            tissues=_tissues,
        )

    with pytest.raises(ValueError):
        qc_comp.batch_effect_qc_metrics(
            cohort_data_dir=_cohort_data_dir,
            qc_cohort_metrics_dir=_cohort_qc_metrics_dir,
            tissues=None,
        )

    qc_comp.batch_effect_qc_metrics(
        cohort_data_dir=_cohort_data_dir,
        qc_cohort_metrics_dir=_cohort_qc_metrics_dir,
        tissues=_tissues,
    )

    for tissue in _tissues:
        for qc_suffix in settings.QC_SUFFIXES:
            assert os.path.exists(_cohort_qc_metrics_dir / f"{tissue}_combined_{qc_suffix}.csv")
