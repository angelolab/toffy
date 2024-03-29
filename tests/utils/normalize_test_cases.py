import numpy as np
import pandas as pd
import pytest

masses = np.arange(5, 15)
channels = ["chan_{}".format(i) for i in range(len(masses))]
panel = pd.DataFrame({"Mass": masses, "Target": channels})
fovs = ["fov{}".format(12 - i) for i in range(5)]


def generate_tuning_data(channel_counts):
    """Creates mph and channel count data frames in the appropriate tuning file format.

    Args:
        channel_counts (np.array): random values, may include low values for testing
    Returns:
        mph_df, count_df (pd.DataFrame), combined dataframes for randomly generated mph and
          provided counts

    """
    # create lists to hold values from each fov in directory
    mph_vals = np.random.randint(1, 200, len(masses))
    fov = np.repeat(1, len(masses))

    # create dfs from current directory
    mph_df = pd.DataFrame({"mass": masses, "fov": fov, "pulse_height": mph_vals})

    # count_df has fields in different order to check that matching is working
    count_df = pd.DataFrame({"mass": masses, "channel_count": channel_counts, "fov": fov})

    return mph_df, count_df


class TuningCurveFiles:
    """Class of various possible tuning curve cases."""

    def case_default_combined_files(self):
        """Default case."""
        dirs = ["Detector_202{}v_2022-01-13_13-30-5{}".format(i, i) for i in range(1, 5)]

        # create lists to hold dfs from each directory
        mph_dfs, count_dfs = [], []

        for dir in dirs:
            # create dfs from current directory
            channel_counts = np.random.randint(3, 100, len(masses))
            mph_df, count_df = generate_tuning_data(channel_counts)

            mph_dfs.append(mph_df)
            count_dfs.append(count_df)

        return dirs, mph_dfs, count_dfs

    @pytest.mark.xfail(raises=ValueError)
    def case_low_count_fail(self):
        """Low enough counts to result in failure."""
        # 5 fovs, 2 with low count fails
        dirs = ["Detector_202{}v_2022-01-13_13-30-5{}".format(i, i) for i in range(1, 6)]

        # create lists to hold dfs from each directory
        mph_dfs, count_dfs = [], []

        for i, dir in enumerate(dirs):
            # include a low channel value in the last fov
            if i == 0 or i == 4:
                channel_counts = np.random.randint(3, 100, len(masses) - 1)
                channel_counts = np.append(channel_counts, 0)
            else:
                channel_counts = np.random.randint(3, 100, len(masses))

            mph_df, count_df = generate_tuning_data(channel_counts)

            mph_dfs.append(mph_df)
            count_dfs.append(count_df)

        return dirs, mph_dfs, count_dfs

    @pytest.mark.filterwarnings("ignore:The counts for the FOV")
    def case_low_count_warn(self):
        """Low counts that trigger a warning."""
        # 6 fovs, 2 with low count passes
        dirs = ["Detector_202{}v_2022-01-13_13-30-5{}".format(i, i) for i in range(1, 7)]

        # create lists to hold dfs from each directory
        mph_dfs, count_dfs = [], []

        for i, dir in enumerate(dirs):
            # include a low channel value in the last fov
            if i == 0 or i == 3:
                channel_counts = np.random.randint(3, 100, len(masses) - 1)
                channel_counts = np.append(channel_counts, 0)
            else:
                channel_counts = np.random.randint(3, 100, len(masses))

            mph_df, count_df = generate_tuning_data(channel_counts)

            mph_dfs.append(mph_df)
            count_dfs.append(count_df)

        return dirs, mph_dfs, count_dfs


class CombineRunMetricFiles:
    """Class of example data."""

    def case_default_metrics(self):
        """Default case."""
        # create full directory of files, include proficient data which should be ignored
        metrics = []
        metrics_prof = []
        for i in range(0, 5):
            metric_name = "pulse_heights_{}.csv".format(i)
            metric_prof_name = "pulse_heights_{}_proficient.csv".format(i)
            metric_values = {
                "pulse_height": np.random.rand(10),
                "mass": masses,
                "fov": [fovs[i]] * 10,
            }
            metric_prof_values = {
                "pulse_height": np.random.rand(10),
                "mass": masses,
                "fov": [fovs[i]] * 10,
            }
            metrics.append([metric_name, metric_values])
            metrics_prof.append([metric_prof_name, metric_prof_values])
        return {"deficient": metrics, "proficient": metrics_prof}
