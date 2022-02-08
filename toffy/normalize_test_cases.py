import pytest

import numpy as np
import pandas as pd

masses = np.arange(5, 20)


class TuningCurveFiles:

    def case_default_combined_files(self):
        dirs = ['dir_{}'.format(i) for i in range(1, 4)]

        # create lists to hold dfs from each directory
        mph_dfs = []
        count_dfs = []

        for dir in dirs:

            # create lists to hold values from each fov in directory
            mph_vals = []
            channel_counts = []
            fovs = []
            num_fovs = np.random.randint(1, 5)

            for i in range(1, num_fovs + 1):
                # initialize random columns for each fov
                mph_vals.extend(np.random.randint(1, 200, len(masses)))
                channel_counts.extend(np.random.randint(3, 100, len(masses)))
                fovs.extend(np.repeat(i, len(masses)))

            # create dfs from current directory
            mph_df = pd.DataFrame({'masses': np.tile(masses, num_fovs),
                                   'fovs': fovs, 'mph': mph_vals})

            # count_df has fields in different order to check that matching is working
            count_df = pd.DataFrame({'masses': np.tile(masses, num_fovs),
                                     'channel_counts': channel_counts, 'fovs': fovs})

            mph_dfs.append(mph_df)
            count_dfs.append(count_df)

        return dirs, mph_dfs, count_dfs


class CombineRunMetricFiles():

    def case_default_metrics(self):
        # create full directory of files
        bins = []
        metrics = []
        for i in range(1, 5):
            bin_name = 'example_{}.bin'.format(i)
            bins.append(bin_name)
            metric_name = 'example_metric_{}.csv'.format(i)
            metric_values = {'column_1': np.random.rand(10),
                             'column_2': np.random.rand(10),
                             'column_3': np.random.rand(10)}
            metrics.append([metric_name, metric_values])
        return bins, metrics