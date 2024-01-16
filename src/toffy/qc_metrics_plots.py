import os
import pathlib
from typing import List, Optional, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from tqdm.auto import tqdm

from toffy import settings
from toffy.qc_comp import QCTMA, QCControlMetrics


def visualize_qc_metrics(
    metric_name: str,
    qc_metric_dir: Union[str, pathlib.Path],
    save_dir: Union[str, pathlib.Path],
    channel_filters: Optional[List[str]] = ["chan_"],
    axes_font_size: int = 16,
    wrap: int = 6,
    dpi: int = 300,
    return_plot: bool = False,
) -> Optional[sns.FacetGrid]:
    """Visualize the barplot of a specific QC metric.

    Args:
        metric_name (str):
            The name of the QC metric to plot. Used as the y-axis label. Options include:
            `"Non-zero mean intensity"`, `"Total intensity"`, `"99.9% intensity value"`.
        qc_metric_dir (Union[str, pathlib.Path]):
            The path to the directory containing the `'combined_{qc_metric}.csv'` files
        save_dir (Optional[Union[str, pathlib.Path]], optional):
            The name of the directory to save the plot to. Defaults to None.
        channel_filters (List[str], optional):
            A list of channels to filter out.
        axes_font_size (int, optional):
            The font size of the axes labels. Defaults to 16.
        wrap (int, optional):
            The number of plots to display per row. Defaults to 6.
        dpi (Optional[int], optional):
            The resolution of the image to use for saving. Defaults to None.
        return_plot (bool):
            If `True`, this will return the plot. Defaults to `False`

    Raises:
        ValueError:
            When an invalid metric is provided.
        FileNotFoundError:
            The QC metric directory `qc_metric_dir` does not exist.
        FileNotFoundError:
            The QC metric `combined_csv` file is does not exist in `qc_metric_dir`.

    Returns:
        Optional[sns.FacetGrid]: Returns the Seaborn FacetGrid catplot of the QC metrics.
    """
    # verify the metric provided is valid
    if metric_name not in settings.QC_COLUMNS:
        raise ValueError(
            "Invalid metric %s provided, must be set to 'Non-zero mean intensity', "
            "'Total intensity', or '99.9%% intensity value'" % metric_name
        )

    # verify the path to the QC metric datasets exist
    if not os.path.exists(qc_metric_dir):
        raise FileNotFoundError("qc_metric_dir %s does not exist" % qc_metric_dir)

    # get the file name of the combined QC metric .csv file to use
    qc_metric_index = settings.QC_COLUMNS.index(metric_name)
    qc_metric_suffix = settings.QC_SUFFIXES[qc_metric_index]
    qc_metric_path = os.path.join(qc_metric_dir, "combined_%s.csv" % qc_metric_suffix)

    # ensure the user set the right qc_metric_dir
    if not os.path.exists(qc_metric_path):
        raise FileNotFoundError(
            "Could not locate %s, ensure qc_metric_dir is correct" % qc_metric_path
        )

    # read in the QC metric data
    qc_metric_df = pd.read_csv(qc_metric_path)

    # filter out naturally-occurring elements as well as Noodle
    qc_metric_df = qc_metric_df[~qc_metric_df["channel"].isin(settings.QC_CHANNEL_IGNORE)]

    # filter out any channel in the channel_filters list
    if channel_filters is not None:
        qc_metric_df: pd.DataFrame = qc_metric_df[
            ~qc_metric_df["channel"].str.contains("|".join(channel_filters))
        ]

    # catplot allows for easy facets on a barplot
    qc_fg: sns.FacetGrid = sns.catplot(
        x="fov",
        y=metric_name,
        col="channel",
        col_wrap=wrap,
        data=qc_metric_df,
        kind="bar",
        color="black",
        sharex=True,
        sharey=False,
    )

    # remove the 'channel =' in each subplot title
    qc_fg.set_titles(template="{col_name}")
    qc_fg.figure.supxlabel(t="fov", x=0.5, y=0, ha="center", size=axes_font_size)
    qc_fg.figure.supylabel(t=f"{metric_name}", x=0, y=0.5, va="center", size=axes_font_size)
    qc_fg.set(xticks=[])

    # per Erin's visualization remove the default axis title on the y-axis
    # and instead show 'fov' along x-axis and the metric name along the y-axis (overarching)
    qc_fg.set_axis_labels(x_var="", y_var="")
    qc_fg.set_xticklabels([])

    # save the figure always
    # Return the figure if specified.
    qc_fg.savefig(os.path.join(save_dir, f"{metric_name}_barplot_stats.png"), dpi=dpi)

    if return_plot:
        return qc_fg


def qc_tmas_metrics_plot(
    qc_tmas: QCTMA,
    tmas: List[str],
    save_figure: bool = False,
    dpi: int = 300,
) -> None:
    """Produces the QC TMA metrics plot for a given set of QC metrics applied to a user specified
    TMA. The figures are saved in `qc_tma_metrics_dir/figures`.

    Args:
        qc_tmas (QCTMA): The class which contains the QC TMA data, filepaths, and methods.
        QC matrix.
        tmas (str): The TMAs to plot the QC metrics for.
        save_figure (bool, optional): If `True`, the figure is saved in a subdirectory in the
        `QCTMA.qc_tma_metrics_dir` directory. Defaults to `False`.
        dpi (int, optional): Dots per inch, the resolution of the image. Defaults to 300.
    """
    if save_figure:
        fig_dir: pathlib.Path = pathlib.Path(qc_tmas.metrics_dir) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

    # also plot averages
    tmas.append("cross_TMA_averages")

    with tqdm(total=len(tmas), desc="Plotting QC TMA Metric Ranks", unit="TMAs") as pbar:
        for tma in tmas:
            _qc_tma_metrics_plot(qc_tmas, tma, fig_dir=fig_dir, save_figure=save_figure, dpi=dpi)
            pbar.set_postfix(TMA=tma)
            pbar.update(n=1)


def _qc_tma_metrics_plot(
    qc_tmas: QCTMA,
    tma: str,
    fig_dir: pathlib.Path,
    save_figure: bool = False,
    dpi: int = 300,
) -> None:
    """Produces the QC TMA metrics plot for a given set of QC metrics applied to a user specified
    TMA. The figures are saved in `qc_tma_metrics_dir/figures`.

    Args:
        qc_tmas (QCTMA): The class which contains the QC TMA data, filepaths, and methods.
        tma (str): The TMA to plot the metrics for.
        fig_dir (pathlib.Path) : Path of where to save the plots.
        save_figure (bool, optional): If `True`, the figure is saved in a subdirectory in the
        `qc_tma_metrics_dir` directory. Defaults to `False`.
        dpi (int, optional): Dots per inch, the resolution of the image. Defaults to 300.
    """
    for qc_metric, suffix in zip(qc_tmas.qc_cols, qc_tmas.qc_suffixes):
        qc_tma_data: np.ndarray = qc_tmas.tma_avg_ranks[tma].loc[qc_metric].values

        # Set up the Figure for multiple axes
        fig: Figure = plt.figure(dpi=dpi)
        qc_tma_data = qc_tma_data.round(decimals=2)

        # Heatmap
        _norm = Normalize(vmin=-1, vmax=1)
        ax_heatmap = fig.add_subplot()
        sns.heatmap(
            data=qc_tma_data,
            square=True,
            ax=ax_heatmap,
            linewidths=1,
            linecolor="black",
            annot_kws={"size": 7},
            cbar_kws={"shrink": 0.5},
            annot=True,
            cmap=sns.color_palette(palette="vlag", as_cmap=True),
            norm=_norm,
        )
        # Set ticks
        ax_heatmap.set_xticks(
            ticks=ax_heatmap.get_xticks(),
            labels=[f"{i+1}" for i in range(qc_tma_data.shape[1])],
            rotation=0,
        )

        ax_heatmap.set_yticks(
            ticks=ax_heatmap.get_yticks(),
            labels=[f"{i+1}" for i in range(qc_tma_data.shape[0])],
            rotation=0,
        )

        ax_heatmap.set_xlabel("Column")
        ax_heatmap.set_ylabel("Row")
        if tma == "averages":
            ax_heatmap.set_title("Average Scores Across All TMAs\n")
        else:
            ax_heatmap.set_title(f"{tma} - Average Channel Z-score per FOV\n")

        if save_figure:
            fig.savefig(
                fname=fig_dir / f"{tma}_{suffix}.png",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(fig)


def longitudinal_control_heatmap(
    qc_control: QCControlMetrics,
    control_sample_name: str,
    save_lc_df: bool = False,
    save_figure: bool = False,
    display_figure: bool = False,
    figsize: tuple[int, int] = (12, 12),
    dpi: int = 300,
) -> None:
    """Generates a heatmap of the QC metrics for the QC Control FOVs.

    Args:
        qc_control (QCControlMetrics): The class which contains the QC LC data, filepaths
        , and methods.
        control_sample_name (List[str]): A list of tissues to plot the QC metrics for.
        save_lc_df: (bool, optional): If `True`, the longitudinal control data is saved as a `csv` in
        `qc_control.metrics_dir`. Defaults to `False`.
        save_figure (bool, optional): If `True`, the figure is saved in a subdirectory in the
        `longitudinal_control_metrics_dir` directory. Defaults to `False`.
        display_figure (bool, optional): If `True`, the figure is displayed. Defaults to `False`.
        figsize: (tuple[int, int], optional): The size of the figure. Defaults to (12, 12).
        dpi (int, optional): Dots per inch, the resolution of the image. Defaults to 300.

    Raises:
        ValueError: Raised when the input tissues are not a list of strings.
    """
    if control_sample_name is None or not isinstance(control_sample_name, str):
        raise ValueError("The control sample name must be string.")
    if save_figure:
        fig_dir: pathlib.Path = pathlib.Path(qc_control.metrics_dir) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

    for qc_col, qc_suffix in zip(qc_control.qc_cols, qc_control.qc_suffixes):
        # Try to read the transformed df if it exists
        try:
            t_df: pd.DataFrame = pd.read_csv(
                os.path.join(
                    qc_control.metrics_dir, f"{control_sample_name}_transformed_{qc_suffix}.csv"
                )
            )
            t_df.rename_axis("fov", axis=1, inplace=True)
        # If it doesn't exist, transform the data and save it.
        except FileNotFoundError:
            t_df: pd.DataFrame = qc_control.transformed_control_effects_data(
                control_sample_name=control_sample_name, qc_metric=qc_col, to_csv=save_lc_df
            )

        # Set up the Figure for multiple axes
        fig: Figure = plt.figure(figsize=figsize, dpi=dpi)
        fig.set_layout_engine(layout="constrained")
        gs = gridspec.GridSpec(nrows=2, ncols=1, figure=fig, height_ratios=[len(t_df.index) - 1, 1])

        # Colorbar Normalization
        _norm = Normalize(vmin=-1, vmax=1)
        _cmap = sns.color_palette("vlag", as_cmap=True)

        fig.suptitle(f"{control_sample_name} - QC: {qc_col}")

        # Annontation kwargs
        annotation_kws = {
            "horizontalalignment": "center",
            "verticalalignment": "center",
            "fontsize": 8,
        }

        # Heatmap
        ax_heatmap: Axes = fig.add_subplot(gs[0, 0])

        sns.heatmap(
            t_df[~t_df.index.isin(["mean"])],
            ax=ax_heatmap,
            linewidths=1,
            linecolor="black",
            cbar_kws={"shrink": 0.5},
            annot=True,
            annot_kws=annotation_kws,
            xticklabels=False,
            norm=_norm,
            cmap=_cmap,
        )

        # cbar title
        ax_heatmap.collections[0].colorbar.ax.set_title(r"$\log_2(QC)$")

        # Axes labels, and ticks
        ax_heatmap.set_yticks(
            ticks=ax_heatmap.get_yticks(),
            labels=ax_heatmap.get_yticklabels(),
            rotation=0,
        )
        ax_heatmap.set_xlabel(None)

        # Averaged values
        ax_avg: Axes = fig.add_subplot(gs[1, 0])

        sns.heatmap(
            data=t_df[t_df.index.isin(["mean"])],
            ax=ax_avg,
            linewidths=1,
            linecolor="black",
            annot=True,
            annot_kws=annotation_kws,
            fmt=".2f",
            cmap=ListedColormap(["white"]),
            cbar=False,
        )
        ax_avg.set_yticks(
            ticks=ax_avg.get_yticks(),
            labels=["Mean"],
            rotation=0,
        )
        ax_avg.set_xticks(
            ticks=ax_avg.get_xticks(),
            labels=ax_avg.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax_heatmap.set_ylabel("Channel")
        ax_avg.set_xlabel("FOV")

        # Save figure
        if save_figure:
            fig.savefig(
                fname=pathlib.Path(qc_control.metrics_dir)
                / "figures"
                / f"{control_sample_name}_heatmap_{qc_suffix}.png",
                dpi=dpi,
                bbox_inches="tight",
            )
        if display_figure:
            fig.show()
        else:
            plt.close(fig)
