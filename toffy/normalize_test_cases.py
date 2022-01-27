import pytest

import numpy as np
import pandas as pd


masses = np.arange(5, 20)


class TuningCurveFiles:
    dirs = ['dir_{}'.format(i) for i in range(1, 4)]

    return_data = []

    for dir in dirs:
        mph_vals = np.random.randint(1, 13, len(masses))
        channel_counts = np.random.randint(3, 16, len(masses))
        fovs = np.arange(13)
        # fovs_rev = list(range(14, 1, -1))
        fovs_rev = np.flip(fovs)

        mph_df = pd.DataFrame({'masses': masses, 'fovs': fovs, 'mph': mph_vals})

        count_df = pd.DataFrame({'masses': masses_rev, 'channel_counts': channel_counts,
                                 'fovs': fovs_rev})

        mph_df.to_csv(os.path.join(temp_dir, dir, 'pulse_heights_combined.csv'), index=False)
        count_df.to_csv(os.path.join(temp_dir, dir, 'channel_counts_combined.csv'), index=False)