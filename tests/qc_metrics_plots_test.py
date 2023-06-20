import itertools
import os
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np
import pandas as pd
import pytest
from alpineer import test_utils
from traitlets import Callable

from toffy import qc_metrics_plots, settings
from toffy.qc_comp import QCTMA, QCBatchEffect

from .qc_comp_test import BatchEffectMetricData, QCMetricData, cohort_data, qc_tmas


def test_visualize_qc_metrics(tmp_path: Path):
    # define the channels to use
    chans = ["chan0", "chan1", "chan2"]

    # define the fov names to use for each channel
    fov_batches = [["fov0", "fov1"], ["fov2", "fov3"], ["fov4", "fov5"]]

    # define the supported metrics to iterate over
    metrics = ["Non-zero mean intensity", "Total intensity", "99.9% intensity value"]

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
            os.path.join(tmp_path, "combined_%s" % qc_metric_suffix), index=False
        )

    # pass an invalid metric
    with pytest.raises(ValueError):
        qc_metrics_plots.visualize_qc_metrics(
            metric_name="bad_metric", qc_metric_dir="", save_dir=""
        )

    # pass an invalid qc_metric_dir
    with pytest.raises(FileNotFoundError):
        qc_metrics_plots.visualize_qc_metrics("Non-zero mean intensity", "bad_qc_dir", save_dir="")

    # pass a qc_metric_dir without the combined files
    os.mkdir(os.path.join(tmp_path, "empty_qc_dir"))
    with pytest.raises(FileNotFoundError):
        qc_metrics_plots.visualize_qc_metrics(
            "Non-zero mean intensity", os.path.join(tmp_path, "empty_qc_dir"), save_dir=""
        )

    # now test the visualization process for each metric
    for metric in metrics:
        # test without saving (should raise an error)
        with pytest.raises(TypeError):
            qc_metrics_plots.visualize_qc_metrics(metric, tmp_path)

        # test with saving
        qc_metrics_plots.visualize_qc_metrics(metric, tmp_path, save_dir=tmp_path)
        assert os.path.exists(os.path.join(tmp_path, "%s_barplot_stats.png" % metric))


@pytest.fixture(scope="function")
def qc_tma_data(qc_tmas: QCMetricData) -> Generator[Callable, None, None]:
    """
    A fixture which yields a function which creates the QCTMA class,
    and computes the metrics, and the rank metrics.

    Args:
        qc_tmas (QCMetricData): The fixture which creates the QCMetricData class.

    Yields:
        Generator[Callable, None, None]: Yields a function which creates the QCTMA class,
        and `tmas` and `channel_exclude` are variables which can be passed to the function.
    """

    def _compute_qc_tmas_metrics(
        tmas: List[str],
        channel_exclude: List[str] = None,
    ) -> QCTMA:
        qc_tmas_data = QCTMA(
            qc_metrics=qc_tmas.qc_metrics,
            cohort_path=qc_tmas.cohort_path,
            metrics_dir=qc_tmas.qc_metrics_dir,
        )

        qc_tmas_data.compute_qc_tma_metrics(tmas=tmas)
        qc_tmas_data.qc_tma_metrics_rank(tmas=tmas, channel_exclude=channel_exclude)

        return qc_tmas_data

    yield _compute_qc_tmas_metrics


@pytest.mark.parametrize(
    "_tmas,_channel_exclude",
    [
        (["Project_TMA1"], ["chan0"]),
        (["Project_TMA1"], ["chan0", "chan1"]),
    ],
)
def test_qc_tmas_metrics_plot(
    qc_tma_data: QCTMA, _tmas: List[str], _channel_exclude: List[str]
) -> None:
    qc_tma: QCTMA = qc_tma_data(tmas=_tmas, channel_exclude=_channel_exclude)

    qc_metrics_plots.qc_tmas_metrics_plot(qc_tmas=qc_tma, tmas=_tmas, dpi=30, save_figure=True)

    total_figures = [
        f"{tissue}_{qc}.png" for tissue, qc in itertools.product(_tmas, qc_tma.qc_suffixes)
    ]

    # Assert the existance of the batch effect figures
    for fig in total_figures:
        assert os.path.exists(qc_tma.metrics_dir / "figures" / fig)


@pytest.fixture(scope="function")
def batch_effect_qc_data(cohort_data: BatchEffectMetricData) -> Generator[Callable, None, None]:
    """
    Creates the QCBatchEffect class, and computes the metrics, and then filters out unwanted
        channels.

    Args:
        cohort_data (BatchEffectMetricData): The Fixture which creates the BatchEffectMetricData
            class.

    Yields:
        Generator[Callable, None, None]: A Function which generates the QCBatchEffect object,
        computes the QC metricss, and filters channels..
    """

    def _compute_qc_batch_effects(
        tissues: List[str],
        channel_include: List[str] = None,
        channel_exclude: List[str] = None,
    ) -> QCBatchEffect:
        qc_batch_effects = QCBatchEffect(
            cohort_data_dir=cohort_data.cohort_data_dir,
            qc_cohort_metrics_dir=cohort_data.cohort_metrics_dir,
            qc_metrics=cohort_data.qc_metrics,
        )

        qc_batch_effects.batch_effect_qc_metrics(tissues=tissues)
        qc_batch_effects.batch_effect_filtering(
            tissues=tissues,
            channel_include=channel_include,
            channel_exclude=channel_exclude,
        )
        return qc_batch_effects

    yield _compute_qc_batch_effects


@pytest.mark.parametrize(
    "_tissues,_channel_include,_channel_exclude",
    [
        (["tissue"], ["chan0"], ["chan1"]),
        (["tissue0", "tissue1", "tissue2"], ["chan0", "chan1"], None),
        (["tissue0"], ["chan0"], ["chan1", "chan2"]),
    ],
)
def test_qc_batch_effect_heatmap(
    batch_effect_qc_data: Callable,
    _tissues: List[str],
    _channel_include: Optional[List[str]],
    _channel_exclude: Optional[List[str]],
) -> None:
    _save_figure: bool = True

    qc_batch: QCBatchEffect = batch_effect_qc_data(_tissues, _channel_include, _channel_exclude)

    qc_metrics_plots.qc_batch_effect_heatmap(
        qc_batch=qc_batch,
        tissues=_tissues,
        save_figure=_save_figure,
        dpi=30,
    )

    total_figures: List[str] = [
        f"{tissue}_heatmap_{qc}.png"
        for tissue, qc in itertools.product(_tissues, qc_batch.qc_suffixes)
    ]

    # Assert the existance of the batch effect figures
    for fig in total_figures:
        assert os.path.exists(qc_batch.qc_cohort_metrics_dir / "figures" / fig)
