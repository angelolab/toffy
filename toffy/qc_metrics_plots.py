from ark.utils.load_utils import load_imgs_from_tree, load_imgs_from_dir
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from toffy import qc_comp
from ark.utils import io_utils
import pandas as pd

def load_normal_fovs(data_dir, 
	                 normal_tissues,
                     channels=None,
                     exclude_channels=None,
                     img_sub_folder=None):
    
    normal_samples = io_utils.list_folders(dir_name=data_dir, substrs=normal_tissues)
    data = load_imgs_from_tree(data_dir=data_dir, 
	                    img_sub_folder=img_sub_folder, 
	                    fovs=normal_samples, 
	                    channels=channels)    
    return data

def qc_metrics_plots(data_dir, 
	                 normal_tissues,
                     channels=None,
                     exclude_channels=None,
	                 qc_metric="99.9th_percentile",
	                 fig_dir=None):
    
    normal_samples = io_utils.list_folders(dir_name=data_dir, substrs=normal_tissues)

    # need to make a nice pandas DF with columns: sample, channel, 99.9th value
    for i in range(len(normal_tissues)):
        data = load_normal_fovs(data_dir,
                                normal_tissues=normal_tissues,
                                channels=channels)
        channels = list(data.channels.values)
        if exclude_channels:
            channels = [x for x in channels if x not in exclude_channels]
        
        
        # For each tissue type, get all the samples
        samples = [x for x in normal_samples if normal_tissues[i] in x]
        plotting_df = pd.DataFrame(columns=["sample",'channel',"tma","99.9th_percentile"])
        
        for j in range(len(channels)):
            # Channel is a gene
            qc_metrics_per_channel = []

            for k in range(len(samples)):
                # go through all samples per gene. list should be of length==samples
                tma = [x for x in samples[k].split("_") if "TMA" in x][0]
                qc_metrics_per_channel += [[normal_tissues[i],
                                           channels[j],
                                           tma,
                                           qc_comp.compute_99_9_intensity(data.loc[samples[k], 0:1023, 0:1023, channels[j]])]]
            

            plotting_df = plotting_df.append( pd.DataFrame(qc_metrics_per_channel,columns=plotting_df.columns) )
        plt.figure(figsize=(20,3))
        ax = sns.violinplot(x="channel",y="99.9th_percentile",data=plotting_df,
                            inner=None,scale="width",color="gray")
        ax = sns.swarmplot(x="channel",y="99.9th_percentile",
                           data=plotting_df,
                           edgecolor="black",hue="tma",palette="tab20")
        ax.set_title(normal_tissues[i])
        plt.xticks(rotation = 45)
        if fig_dir:
            plt.savefig(fig_dir+normal_tissues[i]+"_qc.png",dpi=300)
        plt.show()
        plt.close()
        
    return