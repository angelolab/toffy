from ark.utils.load_utils import load_imgs_from_tree, load_imgs_from_dir
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from toffy import qc_comp
from ark.utils import io_utils,test_utils
import pandas as pd
import tempfile
import os
import qc_metrics_plots


def test_call_violin_swarm():
    # generate fake df
    plotting_df = pd.DataFrame(columns=["sample", "channel", "tma", "99.9th_percentile"])
    plotting_df = plotting_df.append(pd.DataFrame([["TMA1_fov1",
                                                    "chan0",
                                                    "TMA1",
                                                    0.5]], columns=plotting_df.columns))

    # Test that file is created.
    with tempfile.TemporaryDirectory() as temp_dir:
        qc_metrics_plots.call_violin_swarm_plot(plotting_df, fig_label="test123", figsize=(20, 3), 
                                                fig_dir=temp_dir)
        assert os.path.exists(temp_dir+"test123"+"_batch_effects.png")

    # Test that no file is created when not passing fig_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        file_number_before = len(os.listdir(temp_dir))
        qc_metrics_plots.call_violin_swarm_plot(plotting_df, fig_label="test123", figsize=(20, 3), 
                                                fig_dir=None)
        file_number_after = len(os.listdir(temp_dir))
        assert file_number_before == file_number_after


def test_make_batch_effect_plot():
    # Testing exclude_channels would require making a separate function in
    # qc_metrics_plots to generate and return the plotting_df
    with tempfile.TemporaryDirectory() as top_level_dir:
        data_dir = os.path.join(top_level_dir, 'data_dir')
        os.makedirs(data_dir)
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=3)
        chans = ["chan0", "chan1", "chan2"]
        fovs = ["TMA1_fov1", "TMA1_fov2", "TMA2_fov1", "TMA2_fov2"]
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            data_dir, fovs, chans, img_shape=(10, 10), fills=True)

        # Test no plot is made when not passing fig dir
        file_number_before = len(os.listdir(data_dir))
        qc_metrics_plots.make_batch_effect_plot(data_dir=data_dir,
                                                normal_tissues=["fov1"],
                                                exclude_channels=None,
                                                img_sub_folder=None,
                                                qc_metric="99.9th_percentile",
                                                fig_dir=None)
        file_number_after = len(os.listdir(data_dir))
        assert file_number_before == file_number_after

        # Test plots are made when passing fig dir
        qc_metrics_plots.make_batch_effect_plot(data_dir=data_dir,
                                                normal_tissues=["fov1", "fov2"],
                                                exclude_channels=None,
                                                img_sub_folder=None,
                                                qc_metric="99.9th_percentile",
                                                fig_dir=data_dir)
        file_number_after = len(os.listdir(data_dir))
        assert os.path.exists(data_dir+"fov1"+"_batch_effects.png")
        assert os.path.exists(data_dir+"fov2"+"_batch_effects.png")

    # test img sub folder
    with tempfile.TemporaryDirectory() as top_level_dir:
        data_dir = os.path.join(top_level_dir, 'data_dir')
        os.makedirs(data_dir)
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=3)
        chans = ["chan0", "chan1", "chan2"]
        fovs = ["TMA1_fov1", "TMA1_fov2", "TMA2_fov1", "TMA2_fov2"]
        img_sub_folder = "sub_folder"
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            data_dir, fovs, chans, img_shape=(10, 10), fills=True, sub_dir=img_sub_folder)
        qc_metrics_plots.make_batch_effect_plot(data_dir=data_dir,
                                                normal_tissues=["fov1", "fov2"],
                                                exclude_channels=None,
                                                img_sub_folder=img_sub_folder,
                                                qc_metric="99.9th_percentile",
                                                fig_dir=data_dir)
        assert os.path.exists(data_dir+"fov1"+"_batch_effects.png")
