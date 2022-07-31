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


def make_batch_effect_plot_test():
	with tempfile.TemporaryDirectory() as top_level_dir:
	    data_dir = os.path.join(top_level_dir, 'data_dir')
	    os.makedirs(data_dir)

	    # make fake data for testing
	    # file locs are the temp files
	    # data_xr is the image data xarray
	    fovs, chans = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=3)
	    chans=["chan0","chan1","chan2"]
	    fovs=["TMA1_fov1","TMA1_fov2","TMA2_fov1","TMA2_fov2"]
	    filelocs, data_xr = test_utils.create_paired_xarray_fovs(
	        data_dir, fovs, chans, img_shape=(10, 10), fills=True)
	    
	    # no errors
	    qc_metrics_plots.make_batch_effect_plot(data_dir=data_dir,
	                           normal_tissues=["fov1"],
	                           exclude_channels=None,
	                           img_sub_folder=None,
	                           qc_metric="99.9th_percentile",
	                           fig_dir=None)
