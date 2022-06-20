import os
import tempfile

import pandas as pd
from pathlib import Path

from toffy import bin_extraction
from ark.utils import io_utils


def test_extract_missing_fovs():
    panel = pd.DataFrame([{
        'Mass': 98,
        'Target': None,
        'Start': 97.5,
        'Stop': 98.5,
    }])

    # test successful extraction
    bin_file_dir = os.path.join(Path(__file__).parent, "data", "tissue")
    with tempfile.TemporaryDirectory() as extraction_dir:
        bin_extraction.extract_missing_fovs(bin_file_dir, extraction_dir,
                                            panel, extract_intensities=False)
        fovs = ['fov-1-scan-1', 'fov-2-scan-1']
        fovs_extracted = io_utils.list_folders(extraction_dir)

        assert fovs == fovs_extracted

    # test that moly fovs are not extracted
    moly_bin_file_dir = os.path.join(Path(__file__).parent, "data", "moly")
    with tempfile.TemporaryDirectory() as tmpdir:
        extraction_dir = tmpdir

        bin_extraction.extract_missing_fovs(moly_bin_file_dir, extraction_dir,
                                            panel, extract_intensities=False)
        fovs_extracted = io_utils.list_folders(extraction_dir)

        assert len(fovs_extracted) == 0
