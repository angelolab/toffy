import itertools
import os
import pathlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List

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
        if warn_overwrite_test:
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
    """Contains misc information for a testing set of QC information such as the tma names, fovs,
    channels, and a DataFrame containing testing data.
    """

    tma_name: str
    fov_count: int
    channel_ignore_count: int
    tma_n_m: np.ndarray
    fovs: List[str]
    channels: List[str]
    qc_df: pd.DataFrame
    cohort_path: pathlib.Path
    qc_metrics_dir: pathlib.Path
    qc_metrics: List[str]


@pytest.fixture(scope="function")
def qc_tmas(
    rng: np.random.Generator, tmp_path: pathlib.Path
) -> Generator[QCMetricData, None, None]:
    """A Fixture which yields a dataclass used containing the QC dataframe with the three metrics,
    the tma name, fovs, channels and the regex search term for RnCm.

    Args:
        rng (np.random.Generator): The random number generator in `conftest.py`.
        tmp_path (pathlib.Path): The temporary path to the test directory.

    Yields:
        Generator[QCMetricData, None, None]: The dataclass containing testing data.
    """
    # Set up the testing data features
    fov_count: int = 5
    channel_ignore_count: int = len(settings.QC_CHANNEL_IGNORE)
    channel_count: int = 3
    _, chan_names = test_utils.gen_fov_chan_names(num_fovs=fov_count, num_chans=channel_count)
    chan_names.extend(settings.QC_CHANNEL_IGNORE)

    total_n_m_options: np.ndarray = np.arange(0, 13)
    tma_name: str = "Project_TMA1"

    tma_n_m: np.ndarray = rng.choice(a=total_n_m_options, size=(fov_count, 2), replace=False)
    fovs: List[str] = [f"Project_TMA1_R{tma[0]}C{tma[1]}" for tma in tma_n_m]

    # Create the directory containing extracted images for a tma
    cohort_path: Path = tmp_path / "tma"
    cohort_path.mkdir(parents=True, exist_ok=True)

    # Create the directory containing the QC metrics data
    qc_metrics_dir: Path = tmp_path / "qc_metrics"
    qc_metrics_dir.mkdir(parents=True, exist_ok=True)

    _ = test_utils.create_paired_xarray_fovs(
        base_dir=cohort_path,
        fov_names=fovs,
        channel_names=chan_names,
        img_shape=(20, 20),
    )

    qc_df: pd.DataFrame = pd.DataFrame(
        data={
            "fov": fovs * (channel_count + channel_ignore_count),
            "channel": chan_names * fov_count,
            **{
                qc_col: rng.random(size=(fov_count * len(chan_names)))
                for qc_col in settings.QC_COLUMNS
            },
            "row": np.tile(A=tma_n_m[:, 0], reps=len(chan_names)),
            "column": np.tile(A=tma_n_m[:, 1], reps=len(chan_names)),
        }
    )

    for qc_col, qc_suffix in zip(settings.QC_COLUMNS, settings.QC_SUFFIXES):
        qc_df[["fov", "channel", qc_col, "row", "column"]].to_csv(
            qc_metrics_dir / f"{tma_name}_combined_{qc_suffix}.csv", index=False
        )

    yield QCMetricData(
        tma_name=tma_name,
        fov_count=fov_count,
        channel_ignore_count=channel_ignore_count,
        tma_n_m=tma_n_m,
        fovs=fovs,
        channels=chan_names,
        qc_df=qc_df,
        cohort_path=cohort_path,
        qc_metrics_dir=qc_metrics_dir,
        qc_metrics=settings.QC_COLUMNS,
    )


@parametrize(
    "_qc_metrics, _qc_suffixes",
    [
        (settings.QC_COLUMNS[:2], settings.QC_SUFFIXES[:2]),
        ([settings.QC_COLUMNS[0]], [settings.QC_SUFFIXES[0]]),
        (settings.QC_COLUMNS, settings.QC_SUFFIXES),
    ],
)
def test_qc_filtering(_qc_metrics, _qc_suffixes):
    qc_cols, qc_suffixes = qc_comp.qc_filtering(qc_metrics=_qc_metrics)

    # Assert correct number of elements
    assert len(qc_cols) == len(qc_suffixes)
    assert len(qc_suffixes) == len(_qc_metrics)

    # Assert correct ordering of elements
    for qc_col, qc_suffix in zip(qc_cols, qc_suffixes):
        assert qc_col in _qc_metrics
        assert qc_suffix in _qc_suffixes

    # Assert uniqueness of elements
    assert set(qc_cols) == set(_qc_metrics)
    assert set(qc_suffixes) == set(_qc_suffixes)


@parametrize(
    "_channel_exclude, _channel_include",
    [
        (None, None),  # Default, only settings.QC_CHANNEL_IGNORE is removed
        (["chan0", "chan1"], None),  # Remove chan_0 and chan_1
        (["chan0"], ["chan1"]),  # Remove chan_0, and only include chan_1 (df with chan_1 only)
        (
            None,
            ["chan0", "chan1"],
        ),  # Only include chan_0 and chan_1 (df with chan_0 and chan_1 only)
        pytest.param(
            ["chan0", "chan1"],
            ["chan0", "chan_"],
            marks=pytest.mark.xfail,
        ),  # Error, both exclude and include channels are the same
    ],
)
def test__channel_filtering(qc_tmas: QCMetricData, _channel_exclude, _channel_include):
    # Filter out channels that are in the ignore list
    qc_df = qc_comp._channel_filtering(
        df=qc_tmas.qc_df, channel_exclude=_channel_exclude, channel_include=_channel_include
    )

    # Assert that the default excluded channels: Au, Fe, Na, Ta, Noodle are not in the dataframe
    assert set(qc_df["channel"]).isdisjoint(set(settings.QC_CHANNEL_IGNORE))

    # Assert that the excluded channels are removed from the dataframe
    assert set(qc_df["channel"]).isdisjoint(set(_channel_exclude) if _channel_exclude else set())

    # Asser that the included channels are included in the dataframe
    assert set(qc_df["channel"]).issuperset(set(_channel_include) if _channel_include else set())


class TestQCTMA:
    """Class to test QC metrics across the TMA."""

    @pytest.fixture(scope="function", autouse=True)
    def _setup(self, qc_tmas: QCMetricData) -> None:
        """Initialize."""
        self.qc_tmas_fixture = qc_tmas

        self.qc_tma = qc_comp.QCTMA(
            qc_metrics=qc_tmas.qc_metrics,
            cohort_path=qc_tmas.cohort_path,
            metrics_dir=qc_tmas.qc_metrics_dir,
        )

    def test__post_init__(self) -> None:
        """Check output metrics."""
        assert self.qc_tma.qc_cols == settings.QC_COLUMNS
        assert self.qc_tma.qc_suffixes == settings.QC_SUFFIXES

        assert self.qc_tma.tma_avg_ranks == {}

    def test__get_r_c(self) -> None:
        """Test retrieved row and column."""
        result = pd.DataFrame()
        result[["R", "C"]]: pd.DataFrame = self.qc_tmas_fixture.qc_df["fov"].apply(
            lambda row: self.qc_tma._get_r_c(row)
        )

        # Make sure the lengths are the same
        assert len(result) == len(self.qc_tmas_fixture.qc_df)

        # Make sure the set of values are the same for Row and Column
        assert set(result["R"]) == set(self.qc_tmas_fixture.tma_n_m[:, 0])
        assert set(result["C"]) == set(self.qc_tmas_fixture.tma_n_m[:, 1])

    def test_compute_qc_tma_metrics(self) -> None:
        """Test computed TMA metrics."""
        self.qc_tma.compute_qc_tma_metrics(tmas=[self.qc_tmas_fixture.tma_name])

        for ms in settings.QC_SUFFIXES:
            all_metric_files: str = io_utils.list_files(
                self.qc_tmas_fixture.qc_metrics_dir, substrs=f"{ms}.csv"
            )

            # Get the combined file
            combined_metric_file: str = next(filter(lambda mf: "combined" in mf, all_metric_files))

            # Filter out combined files
            metric_files: List[str] = list(
                filter(lambda mf: "combined" not in mf, all_metric_files)
            )

            # Make sure that they all have the `tma_name` as the prefix
            assert all(mf.startswith(self.qc_tmas_fixture.tma_name) for mf in all_metric_files)

            # Combined metric file df
            combined_mf_df = pd.read_csv(self.qc_tmas_fixture.qc_metrics_dir / combined_metric_file)

            for mf in metric_files:
                mf_df: pd.DataFrame = pd.read_csv(self.qc_tmas_fixture.qc_metrics_dir / mf)
                # Merge the dataframes together, and check that "fov", "channel", and metric val
                # Assert that all elements in the metric csv exist in the combined metric csv
                pd.testing.assert_frame_equal(
                    left=combined_mf_df.merge(mf_df).iloc[:, 0:3], right=mf_df
                )

    def test__create_r_c_tma_matrix(self) -> None:
        """Test row, column matrix creation."""
        y_size, x_size = np.max(self.qc_tmas_fixture.tma_n_m, axis=0)

        for qc_col in settings.QC_COLUMNS:
            r_c_tma_matrix_df = pd.DataFrame()

            r_c_tma_matrix_df[["rc_matrix"]] = self.qc_tmas_fixture.qc_df.groupby(
                by="channel", sort=True
            ).apply(lambda group: self.qc_tma._create_r_c_tma_matrix(group, x_size, y_size, qc_col))

            # Assert that the shapes are correct.
            for rc_matrix in r_c_tma_matrix_df["rc_matrix"]:
                assert rc_matrix.shape == (x_size, y_size)

    @parametrize(
        "channel_exclude",
        [
            (None),
            (["chan0"]),
            (["chan0", "chan1"]),
        ],
    )
    # test_qc_tma_metrics_compute_qc_tma_metrics
    def test_qc_tma_metrics_rank(self, channel_exclude) -> None:
        """Test TMA rank metrics."""
        self.qc_tma.qc_tma_metrics_rank(
            tmas=[self.qc_tmas_fixture.tma_name], channel_exclude=channel_exclude
        )

        assert self.qc_tma.tma_avg_ranks[self.qc_tmas_fixture.tma_name].shape == (
            3,
            *self.qc_tmas_fixture.tma_n_m.max(0).tolist(),
        )


@dataclass
class BatchEffectMetricData:
    """Contains misc information for a testing set of QC information such as the tma names, fovs,
    channels, and a DataFrame containing testing data.
    """

    qc_metrics: List[str]
    qc_df: pd.DataFrame
    fovs: List[str]
    cohort_img_dir: pathlib.Path
    cohort_metrics_dir: pathlib.Path


@pytest.fixture(scope="function")
def cohort_data(
    rng: np.random.Generator, tmp_path: Path
) -> Generator[BatchEffectMetricData, None, None]:
    """A fixture for generating cohort fovs, and channels for various tissues.

    Args:
        rng (np.random.Generator): The random number generator in `conftest.py`.
        tmp_path (Path): A temporary directory to write files for testing.

    Yields:
        Generator[BatchEffectMetricData, None, None]: Yields a dataclass containing
        the testing data: qc_metrics, qc dataframe, tissues, and cohort data_directory.
    """
    # Set up Directories for the cohort data
    cohort_dir: Path = tmp_path / "my_cohort"
    cohort_img_dir: Path = cohort_dir / "images"
    cohort_metrics_dir: Path = cohort_dir / "metrics"

    for directory in [cohort_dir, cohort_img_dir, cohort_metrics_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Set up the testing data fovs
    channel_count: int = 5
    fov_count: int = 3

    fov_names, chan_names = test_utils.gen_fov_chan_names(
        num_fovs=fov_count, num_chans=channel_count
    )

    # Add the default channels to ignore
    chan_names.extend(settings.QC_CHANNEL_IGNORE)

    _ = test_utils.create_paired_xarray_fovs(
        base_dir=cohort_img_dir, fov_names=fov_names, channel_names=chan_names, img_shape=(20, 20)
    )

    control_sample_names = ["batch0", "batch1", "batch2", "batch3"]

    qc_df: pd.DataFrame = pd.DataFrame(
        data={
            "fov": fov_names * (channel_count + len(settings.QC_CHANNEL_IGNORE)),
            "channel": chan_names * fov_count,
            **{
                qc_col: rng.random(
                    size=(fov_count * (channel_count + len(settings.QC_CHANNEL_IGNORE)))
                )
                for qc_col in settings.QC_COLUMNS
            },
        }
    )

    for control_sample_name, (qc_col, qc_suffix) in itertools.product(
        control_sample_names, zip(settings.QC_COLUMNS, settings.QC_SUFFIXES)
    ):
        qc_df[
            [
                "fov",
                "channel",
                qc_col,
            ]
        ].to_csv(
            cohort_metrics_dir / f"{control_sample_name}_combined_{qc_suffix}.csv", index=False
        )

    yield BatchEffectMetricData(
        qc_metrics=settings.QC_COLUMNS,
        qc_df=qc_df,
        fovs=fov_names,
        cohort_img_dir=cohort_img_dir,
        cohort_metrics_dir=cohort_metrics_dir,
    )


class TestQCControlMetrics:
    """Class to test QC metrics across the controls."""

    @pytest.fixture(autouse=True)
    def _setup(self, cohort_data: BatchEffectMetricData) -> None:
        """Initialize."""
        self.cohort_data: BatchEffectMetricData = cohort_data

        self.qc_control_metrics = qc_comp.QCControlMetrics(
            qc_metrics=settings.QC_COLUMNS,
            cohort_path=self.cohort_data.cohort_img_dir,
            metrics_dir=self.cohort_data.cohort_metrics_dir,
        )

    def test__post_init__(self) -> None:
        """Check output metrics."""
        assert self.qc_control_metrics.qc_cols == settings.QC_COLUMNS
        assert self.qc_control_metrics.qc_suffixes == settings.QC_SUFFIXES

        assert self.qc_control_metrics.longitudinal_control_metrics == {}

    @parametrize(
        "_control_sample_name,_channel_include,_channel_exclude",
        [
            ("batch0", None, None),
            ("batch1", ["chan0"], None),
            ("batch2", None, ["chan0", "chan1"]),
            pytest.param("batch3", ["chan0"], ["chan0"], marks=pytest.mark.xfail),
            pytest.param(None, None, None, marks=pytest.mark.xfail),
        ],
    )
    def test_compute_control_qc_metrics(
        self,
        _control_sample_name,
        _channel_include,
        _channel_exclude,
    ) -> None:
        """Test metrics computation."""
        self.qc_control_metrics.compute_control_qc_metrics(
            control_sample_name=_control_sample_name,
            fovs=self.cohort_data.fovs,
            channel_exclude=_channel_exclude,
            channel_include=_channel_include,
        )

        for qc_suffix in settings.QC_SUFFIXES:
            assert os.path.exists(
                self.qc_control_metrics.metrics_dir
                / f"{_control_sample_name}_combined_{qc_suffix}.csv"
            )

            qc_df = pd.read_csv(
                self.qc_control_metrics.metrics_dir
                / f"{_control_sample_name}_combined_{qc_suffix}.csv"
            )
            assert set(qc_df["channel"]).isdisjoint(set(settings.QC_CHANNEL_IGNORE))

            # Assert that the excluded channels are removed from the dataframe
            assert set(qc_df["channel"]).isdisjoint(
                set(_channel_exclude) if _channel_exclude else set()
            )

            # Assert that the included channels are included in the dataframe
            assert set(qc_df["channel"]).issuperset(
                set(_channel_include) if _channel_include else set()
            )

    @parametrize(
        "_control_sample_name,_metric",
        [
            ("batch0", settings.QC_COLUMNS[0]),
            ("batch1", settings.QC_COLUMNS[1]),
            ("batch2", settings.QC_COLUMNS[2]),
            (
                "batch3",
                settings.QC_COLUMNS[0],
            ),  # Test reading from a file if the metric has been already computed
        ],
    )
    def test_transformed_control_effects_data(self, _control_sample_name, _metric) -> None:
        """Test tranformed data."""
        self.qc_control_metrics.compute_control_qc_metrics(
            control_sample_name=_control_sample_name,
            fovs=self.cohort_data.fovs,
            channel_exclude=None,
            channel_include=None,
        )

        transformed_df = self.qc_control_metrics.transformed_control_effects_data(
            control_sample_name=_control_sample_name, qc_metric=_metric, to_csv=True
        )

        # Asser that the transformed control effects file is created.
        _qc_suffix = self.qc_control_metrics.qc_suffixes[
            self.qc_control_metrics.qc_cols.index(_metric)
        ]
        transformed_df_path = os.path.join(
            self.qc_control_metrics.metrics_dir,
            f"{_control_sample_name}_transformed_{_qc_suffix}.csv",
        )

        assert os.path.exists(transformed_df_path)
        # Assert that the transformed csv is the same as the one returned by the function

        saved_transformed_df = pd.read_csv(transformed_df_path, index_col=["channel"])
        saved_transformed_df.rename_axis("fov", axis=1, inplace=True)

        pd.testing.assert_frame_equal(left=transformed_df, right=saved_transformed_df)

        # All FOVs exist
        assert set(transformed_df.axes[1]) == set(self.cohort_data.qc_df["fov"])

        # Ignored channels are not in the dataframe
        assert set(transformed_df.axes[0]).isdisjoint(set(settings.QC_CHANNEL_IGNORE))
