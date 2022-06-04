import pytest

import numpy as np
import pandas as pd

masses = np.arange(5, 15)
channels = ['chan_{}'.format(i) for i in range(len(masses))]
panel = pd.DataFrame({'Mass': masses, 'Target': channels})


class TuningCurveFiles:

    def case_default_combined_files(self):
        dirs = ['Detector_202{}v_2022-01-13_13-30-5{}'.format(i, i) for i in range(1, 4)]

        # create lists to hold dfs from each directory
        mph_dfs = []
        count_dfs = []

        for dir in dirs:

            # create lists to hold values from each fov in directory
            mph_vals = np.random.randint(1, 200, len(masses))
            channel_counts = np.random.randint(3, 100, len(masses))
            fovs = np.repeat(1, len(masses))

            # create dfs from current directory
            mph_df = pd.DataFrame({'mass': masses, 'fov': fovs, 'pulse_height': mph_vals})

            # count_df has fields in different order to check that matching is working
            count_df = pd.DataFrame({'mass': masses, 'channel_count': channel_counts, 'fov': fovs})

            mph_dfs.append(mph_df)
            count_dfs.append(count_df)

        return dirs, mph_dfs, count_dfs


class CombineRunMetricFiles:

    def case_default_metrics(self):
        # create full directory of files
        metrics = []
        for i in range(1, 5):
            metric_name = 'pulse_heights_{}.csv'.format(i)
            metric_values = {'pulse_height': np.random.rand(10),
                             'mass': masses,
                             'fov': ['fov{}'.format(12 - i)] * 10}
            metrics.append([metric_name, metric_values])
        return metrics
