import pandas as pd

import pytest


class CompensateImageDataPanel:

    def case_sorted_panel(self):
        masses = ['25', '50', '101']
        chans = ['chan0', 'chan1', 'chan2']

        d = {'masses': masses, 'targets': chans}
        panel_info = pd.DataFrame(d)
        return panel_info

    @pytest.mark.xfail
    def case_unsorted_panel(self):
        masses = ['25', '50', '49']
        chans = ['chan0', 'chan1', 'chan2']

        d = {'masses': masses, 'targets': chans}
        panel_info = pd.DataFrame(d)
        return panel_info
