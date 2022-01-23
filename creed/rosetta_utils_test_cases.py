import numpy as np
import pandas as pd


import pytest

xfail = pytest.mark.xfail


class CompensateImageDataPanel:
    # Create panel info pandas array

    def case_sorted_panel(self):
        # masses are sorted in numerical order, as expected
        masses = ['25', '50', '101']
        chans = ['chan0', 'chan1', 'chan2']

        d = {'masses': masses, 'targets': chans}
        panel_info = pd.DataFrame(d)
        return panel_info

    @xfail(raises=ValueError, strict=True)
    def case_unsorted_panel(self):
        # masses are not sorted
        masses = ['25', '50', '49']
        chans = ['chan0', 'chan1', 'chan2']

        d = {'masses': masses, 'targets': chans}
        panel_info = pd.DataFrame(d)
        return panel_info


class CompensateImageDataMat:
    # create image compensation matrix

    def case_matching_channels(self):
        # include same masses as panel_info
        comp_mat_vals = np.random.rand(3, 3) / 100
        comp_mat = pd.DataFrame(comp_mat_vals, columns=['25', '50', '101'],
                                index=['25', '50', '101'])
        return comp_mat

    @xfail(raises=ValueError, strict=True)
    def case_non_matching_channels(self):
        # include different masses from panel_info
        comp_mat_vals = np.random.rand(3, 3) / 100
        comp_mat = pd.DataFrame(comp_mat_vals, columns=['25', '50', '102'],
                                index=['25', '50', '102'])
        return comp_mat
