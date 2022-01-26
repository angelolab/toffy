import numpy as np
import pandas as pd


import pytest

xfail = pytest.mark.xfail

# defaults for creating test data
test_mass_list = ['25', '50', '101']
test_chans_list = ['chan0', 'chan1', 'chan2']


class CompensateImageDataPanel:
    # Create panel info pandas array

    def case_sorted_panel(self):
        # masses are sorted in numerical order, as expected
        d = {'masses': test_mass_list, 'targets': test_chans_list}
        panel_info = pd.DataFrame(d)
        return panel_info

    @xfail(raises=ValueError, strict=True)
    def case_unsorted_panel(self):
        # masses are not sorted
        masses = ['25', '101', '50']

        d = {'masses': masses, 'targets': test_chans_list}
        panel_info = pd.DataFrame(d)
        return panel_info


class CompensateImageDataMat:
    # create image compensation matrix

    def case_matching_channels(self):
        # include same masses as panel_info
        comp_mat_vals = np.random.rand(len(test_mass_list), len(test_mass_list)) / 100
        comp_mat = pd.DataFrame(comp_mat_vals, columns=test_mass_list, index=test_mass_list)
        return comp_mat

    @xfail(raises=ValueError, strict=True)
    def case_non_matching_channels(self):
        # include different masses from panel_info
        comp_mat_vals = np.random.rand(len(test_mass_list), len(test_mass_list)) / 100
        different_masses = ['25', '50', '102']
        comp_mat = pd.DataFrame(comp_mat_vals, columns=different_masses, index=different_masses)
        return comp_mat
