import itertools
import os
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import natsort as ns
import numpy as np
import pandas as pd
import seaborn as sns
from alpineer import io_utils, misc_utils
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from toffy import settings


def batch_effect_plot(
    qc_cohort_metrics_dir: str | Path,
    tissues: list[str],
    qc_metrics: list[str] | None = None,
    channel_include: list[str] | None = None,
    channel_exclude: list[str] | None = None,
    save_figure: bool = False,
):
    """
    Creates a combination violin and swarm plot in order to compare QC metrics across channels
    for a given set of TMAs. Undesired channels can be filtered out and / or strictly desired channels
    can included exclusively.

    Args:
        qc_cohort_metrics_dir (str | Path): The directory where the cohort metrics will be saved to.
        tissues (list[str] | None, optional): A list of tissues to plot.
        qc_metrics (list[str] | None, optional): A list of QC metrics to plot for each tissue.
        Defaults to None, where all QC metrics are plotted.
        channel_include (list[str] | None, optional): A list of channels to include in the plots. Defaults to None.
        channel_exclude (list[str] | None, optional): A list of channels to exclude in the plots. Defaults to None.
        save_figure (bool, optional): If `True`, the figure is saved in a subdirectory in the
        `qc_cohort_metrics_dir` directory. Defaults to `False`.

    Raises:
        ValueError: Raised when channel_include and channel_exclude contain a shared channel.
    """
    # Input validation

    if isinstance(channel_include, list) and isinstance(channel_exclude, list):
        if set(channel_exclude).isdisjoint(set(channel_include)):
            raise ValueError("You cannot include and exlude the same channel.")

    # Filter out unused QC suffixess
    if qc_metrics is not None:
        selected_qcs: list[bool] = [qcm in qc_metrics for qcm in settings.QC_COLUMNS]
        qc_cols = list(itertools.compress(settings.QC_COLUMNS, selected_qcs))
        qc_suffixes = list(itertools.compress(settings.QC_SUFFIXES, selected_qcs))
    else:
        qc_cols: list[str] = settings.QC_COLUMNS
        qc_suffixes: list[str] = settings.QC_SUFFIXES

    # A dictionary s.t. {tisue1: [list of qc files, sorted by qc_suffixes],
    #                   tissue2: [list of qc files, sorted by qc_suffixes],
    #                   ...                                             }
    combined_batches: dict[str, list[str]] = {
        tissue: ns.natsorted(
            io_utils.list_files(
                qc_cohort_metrics_dir,
                substrs=[f"{tissue}_combined_{qc_suffix}" for qc_suffix in qc_suffixes],
            ),
            key=lambda m: (i for i, qc_s in enumerate(qc_suffixes) if qc_s in m),
        )
        for tissue in tissues
    }

    for tissue, tissue_stats in combined_batches.items():
        for tissue_stat, qc_col, qc_suffix in zip(tissue_stats, qc_cols, qc_suffixes):
            # Load the csv and filter the default ignored channels
            be_df: pd.DataFrame = pd.read_csv(os.path.join(qc_cohort_metrics_dir, tissue_stat))
            be_df: pd.DataFrame = be_df[~be_df["channel"].isin(settings.QC_CHANNEL_IGNORE)]

            # Verify that the excluded channels exist in the combined metric Tissue DataFrame
            # Then remove the excluded channels
            if channel_exclude is not None:
                misc_utils.verify_in_list(
                    channels_to_exclude=channel_exclude,
                    combined_tissue_df_channels=be_df["channel"].unique(),
                )
                be_df: pd.DataFrame = be_df[~be_df["channel"].isin(channel_exclude)]

            # Verify that the included channels exist in the combined metric Tissue DataFrame
            # Then filter the excluded channels
            if channel_include is not None:
                misc_utils.verify_in_list(
                    channels_to_include=channel_include,
                    combined_tissue_df_channels=be_df["channel"].unique(),
                )
                be_df: pd.DataFrame = be_df[be_df["channel"].isin(channel_include)]

            # Set up the Figure for one axes (which gets overlayed)
            fig: Figure = plt.figure(figsize=(max(be_df["channel"].nunique() // 4, 4), 6), dpi=600)
            fig.set_layout_engine(layout="constrained")
            gs = gridspec.GridSpec(1, 1, figure=fig)
            fig.suptitle(f"{tissue} - {qc_col}")

            # Create shared axis object and plot violin and swarm
            violin_swarm_batch_effect_ax: Axes = fig.add_subplot(gs[0, 0])

            sns.violinplot(
                data=be_df,
                x="channel",
                y=qc_col,
                ax=violin_swarm_batch_effect_ax,
                inner=None,
                scale="width",
                color="lightgrey",
                linewidth=1,
            )

            sns.swarmplot(
                data=be_df,
                x="channel",
                y=qc_col,
                ax=violin_swarm_batch_effect_ax,
                edgecolor="black",
                hue="fov",
                size=3,
            )

            # Post plotting adjustments

            # x axis
            violin_swarm_batch_effect_ax.set_xticks(
                ticks=violin_swarm_batch_effect_ax.get_xticks(),
                labels=violin_swarm_batch_effect_ax.get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )
            violin_swarm_batch_effect_ax.set_xlabel("Channel")

            # Legend

            # Sort legend labels
            _handles, _labels = violin_swarm_batch_effect_ax.get_legend_handles_labels()
            legend_labels_index: list = ns.index_natsorted(_labels)
            ordered_handles: list = ns.order_by_index(_handles, legend_labels_index)
            ordered_labels: list = ns.order_by_index(_labels, legend_labels_index)

            sns.move_legend(
                violin_swarm_batch_effect_ax,
                loc="best",
                title="FOVs",
                fontsize="x-small",
                handles=ordered_handles,
                labels=ordered_labels,
            )

            # Save figure
            if save_figure:
                fig_dir: Path = Path(qc_cohort_metrics_dir) / "figures"
                fig_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(fname=fig_dir / f"{tissue}_{qc_suffix}")


def qc_tma_metrics_plot(
    cmt_data: dict[str, np.ndarray],
    qc_tma_metrics_dir: str | Path,
    tma: str,
    save_figure: bool = False,
) -> None:
    """
    Produces the QC TMA metrics plot for a given set of QC metrics applied to a user specified
    TMA. The figures are saved in `qc_tma_metrics_dir/figures`. By default the following channels
    are excluded: Au, Fe, Na, Ta, Noodle.


    Args:
        cmt_data (dict[str, np.ndarray]): The dictionary of the QC metrics and the associated TMA
        QC matrix.
        qc_tma_metrics_dir (str | Path): The directory where to place the QC TMA metrics.
        tma (str): The FOVs with the TMA in the folder name to gather.
        save_figure (bool, optional): If `True`, the figure is saved in a subdirectory in the
        `qc_tma_metrics_dir` directory. Defaults to `False`.

    """

    # Filter QC columns, and file suffixes
    # makes the list of qc_cols and qc_suffixes ordered the same.
    filtered_qcs: list[bool] = [qcm in cmt_data.keys() for qcm in settings.QC_COLUMNS]
    qc_cols = list(itertools.compress(settings.QC_COLUMNS, filtered_qcs))
    qc_suffixes = list(itertools.compress(settings.QC_SUFFIXES, filtered_qcs))

    for qc_metric, suffix in zip(qc_cols, qc_suffixes):
        # Set up the Figure for multiple axes
        fig: Figure = plt.figure()
        fig.set_layout_engine(layout="constrained")
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.1)
        fig.suptitle(f"{tma} - {qc_metric}")

        # Heatmap
        ax_heatmap: Axes = fig.add_subplot(gs[0, 0])
        sns.heatmap(
            cmt_data[qc_metric],
            square=True,
            ax=ax_heatmap,
            linewidths=1,
            linecolor="black",
            cbar_kws={"shrink": 0.5},
            annot=True,
        )

        # Set ticks
        ax_heatmap.set_xticks(
            ticks=ax_heatmap.get_xticks(),
            labels=[f"{i+1}" for i in range(cmt_data[qc_metric].shape[0])],
            rotation=0,
        )

        ax_heatmap.set_yticks(
            ticks=ax_heatmap.get_yticks(),
            labels=[f"{i+1}" for i in range(cmt_data[qc_metric].shape[1])],
            rotation=0,
        )

        ax_heatmap.set_xlabel("Column")
        ax_heatmap.set_ylabel("Row")
        ax_heatmap.set_title("Average Rank")

        # Histogram
        ax_hist: Axes = fig.add_subplot(gs[0, 1])
        sns.histplot(cmt_data[qc_metric].ravel(), ax=ax_hist, bins=10)
        ax_hist.set(xlabel="Average Rank", ylabel="Count")
        ax_hist.set_title("Average Rank Distribution")

        if save_figure:
            fig_dir: Path = Path(qc_tma_metrics_dir) / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(fname=fig_dir / f"{tma}_{suffix}.png")
