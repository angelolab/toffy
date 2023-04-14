import itertools
import os
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from alpineer import test_utils

from toffy import qc_metrics_plots, settings


@pytest.fixture(scope="function")
def batch_effect_qc_data(rng: np.random.Generator, tmp_path: Path) -> Generator[Path, None, None]:
    """
    A fixture which saves combined QC metrics for the following test tissues in an imaginary
    cohort: "ln", "ln_top", "ln_bottom". Yields the directory where the cohort's QC metrics are
    saved.

    Args:
        rng (np.random.Generator): The random number generator in `conftest.py`.
        tmp_path (Path): A temporary directory to write files for testing.

    Yields:
        Generator[Path, None, None]: The QC cohort directory.
    """
    qc_cohort_metrics_dir = tmp_path / "cohort_metrics"
    qc_cohort_metrics_dir.mkdir(parents=True, exist_ok=True)

    # Channels: 10 channels as well as the settings.QC_CHANNEL_IGNORE channels
    fovs, channels = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=10)
    channels.extend(settings.QC_CHANNEL_IGNORE)
    channel_count: int = len(channels)

    tissues: list[str] = ["ln", "ln_top", "ln_bottom"]

    for qc_col, qc_suffix in zip(settings.QC_COLUMNS, settings.QC_SUFFIXES):
        for tissue in tissues:
            df = pd.DataFrame(
                data={
                    "fov": [f"{fov}_{tissue}" for fov in fovs] * channel_count,
                    "channel": channels * len(fovs),
                    qc_col: rng.random(size=(len(fovs) * channel_count,)),
                }
            )
            df.to_csv(qc_cohort_metrics_dir / f"{tissue}_combined_{qc_suffix}.csv")

    yield qc_cohort_metrics_dir


# Set various combinations of tissues and qc_metrics
@pytest.mark.parametrize(
    "_tissues,_qc_metrics,_qc_suffixes",
    [
        (["ln"], [settings.QC_COLUMNS[0]], [settings.QC_SUFFIXES[0]]),
        (["ln", "ln_top", "ln_bottom"], settings.QC_COLUMNS, settings.QC_SUFFIXES),
        (["ln_top"], None, settings.QC_SUFFIXES),
    ],
)
# Set various channel include / exclude parameterrs
@pytest.mark.parametrize(
    "_channel_include, _channel_exclude",
    [
        (None, None),  # All channels + default exclude channels
        (["chan0", "chan1", "chan2"], None),  # Only use channels 0-3
        (None, ["chan3", "chan4", "chan5"]),  # Exclude channels 3-5
        pytest.param(
            ["chan0", "chan3"], ["chan1", "chan3"], marks=pytest.mark.xfail
        ),  # inlude and exclude contain a shared channel, will error.
    ],
)
def test_batch_effect_plot(
    batch_effect_qc_data: Path,
    _tissues: list[str],
    _qc_metrics: list[str],
    _qc_suffixes: list[str],
    _channel_include: list[str] | None,
    _channel_exclude: list[str] | None,
):
    _save_figure: bool = True

    qc_metrics_plots.batch_effect_plot(
        qc_cohort_metrics_dir=batch_effect_qc_data,
        tissues=_tissues,
        qc_metrics=_qc_metrics,
        channel_include=_channel_include,
        channel_exclude=_channel_exclude,
        save_figure=_save_figure,
    )

    total_figures = [
        f"{tissue}_{qc}.png" for tissue, qc in itertools.product(_tissues, _qc_suffixes)
    ]

    # Assert the existance of the batch effect figures
    for fig in total_figures:
        assert os.path.exists(batch_effect_qc_data / "figures" / fig)


@pytest.fixture(scope="function")
def cmt_data(rng: np.random.Generator) -> Generator[dict[str, np.ndarray], None, None]:
    """
    A fixture which yields a dictionary of the QC Column, and the channel metric TMA
    ranked averages.

    Args:
        rng (np.random.Generator): The random number generator in `conftest.py`.

    Yields:
        Generator[dict[str, np.ndarray], None, None]: The key consists of the QC Columns, while
        the values are 12 x 12 arrays with a few non-nan floating point values.
    """
    fov_rc: np.ndarray = rng.integers(low=0, high=12, size=(10, 2))
    cmt_dict = {}

    for qc_col in settings.QC_COLUMNS:
        tma_matrix: np.ndarray = np.empty(shape=(12, 12))
        tma_matrix.fill(np.nan)
        for idx in fov_rc:
            x, y = idx
            tma_matrix[x, y] = rng.random(size=1)
        cmt_dict[qc_col] = tma_matrix

    yield cmt_dict


def test_qc_tma_metrics_plot(cmt_data: dict[str, np.ndarray], tmp_path: Path):
    qc_tma_metrics_dir: Path = tmp_path / "metrics"
    qc_tma_metrics_dir.mkdir(parents=True, exist_ok=True)
    tma: str = "TESTING_TMA"

    qc_metrics_plots.qc_tma_metrics_plot(
        cmt_data=cmt_data, qc_tma_metrics_dir=qc_tma_metrics_dir, tma=tma, save_figure=True
    )
    # Assert existance of the tma metrics plots
    for qc_suffix in settings.QC_SUFFIXES:
        assert os.path.exists(qc_tma_metrics_dir / "figures" / f"{tma}_{qc_suffix}.png")
