import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tmi import io_utils, load_utils

from toffy import qc_comp


def call_violin_swarm_plot(plotting_df, fig_label, figsize=(20, 3), fig_dir=None):
    """Makes violin plot with swarm dots. Used with make_batch_effect_plot()

    Args: plotting_df (pandas dataframe): "sample", "channel", "tma", "99.9th_percentile"
          figsize (tuple): (length x width) of figsize
          fig_dir (str): Dir to save plots.
    """
    plt.figure(figsize=figsize)
    ax = sns.violinplot(x="channel", y="99.9th_percentile", data=plotting_df,
                        inner=None, scale="width", color="gray")
    ax = sns.swarmplot(x="channel", y="99.9th_percentile", data=plotting_df,
                       edgecolor="black", hue="tma", palette="tab20")
    ax.set_title(fig_label)
    plt.xticks(rotation=45)
    if fig_dir:
        plt.savefig(fig_dir+fig_label+"_batch_effects.png", dpi=300)
    return ax


def make_batch_effect_plot(data_dir, normal_tissues, exclude_channels=None,
                           img_sub_folder=None, qc_metric="99.9th_percentile", fig_dir=None):
    """Makes violin plots based on tissue type. Calls call_violin_swarm_plot.

    Args:
        normal_tissues (str): is a list of the tissue type substring to match
        exclude_channels (str): is a list of channels to not plot
        img_sub_folder (str): in case theres additional sub folder structure
        qc_metric (str): Type of qc_metric. Currently only 99.9th percentile.

    """
    for i in range(len(normal_tissues)):
        samples = io_utils.list_folders(dir_name=data_dir, substrs=normal_tissues[i])
        data = load_utils.load_imgs_from_tree(data_dir=data_dir,
                                              img_sub_folder=img_sub_folder,
                                              fovs=samples)
        channels = list(data.channels.values)
        if exclude_channels:
            channels = [x for x in channels if x not in exclude_channels]

        # i could add a separate function to produce the plotting_df that is testable
        plotting_df = pd.DataFrame(columns=["sample", "channel", "tma", "99.9th_percentile"])

        for j in range(len(channels)):
            qc_metrics_per_channel = []

            for k in range(len(samples)):
                tma = [x for x in samples[k].split("_") if "TMA" in x][0]
                qc_metrics_per_channel += [[normal_tissues[i],
                                           channels[j],
                                           tma,
                                           qc_comp.compute_99_9_intensity(data.loc[samples[k],
                                                                          :,
                                                                          :,
                                                                          channels[j]])]]

            plotting_df = plotting_df.append(pd.DataFrame(qc_metrics_per_channel,
                                                          columns=plotting_df.columns))

        call_violin_swarm_plot(plotting_df, fig_label=normal_tissues[i], fig_dir=fig_dir)
