import os
import tempfile
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, call

from toffy import bin_extraction
from mibi_bin_tools import io_utils


@patch('builtins.print')
def test_extract_missing_fovs(mocked_print):
    bin_file_dir = os.path.join(Path(__file__).parent, "data", "tissue")
    panel = pd.DataFrame([{
        'Mass': 98,
        'Target': None,
        'Start': 97.5,
        'Stop': 98.5,
    }])

    # test that it does not re-extract fovs
    with tempfile.TemporaryDirectory() as extraction_dir:
        os.makedirs(os.path.join(extraction_dir, 'fov-1-scan-1'))
        bin_extraction.extract_missing_fovs(bin_file_dir, extraction_dir,
                                            panel, extract_intensities=False)
        assert mocked_print.mock_calls == [call('Previous extracted FOVs: ', ['fov-1-scan-1']),
                                           call('Moly FOVs which will not be extracted: ', [])]

    # test successful extraction fovs
    with tempfile.TemporaryDirectory() as extraction_dir:
        bin_extraction.extract_missing_fovs(bin_file_dir, extraction_dir,
                                            panel, extract_intensities=False)
        fovs = ['fov-1-scan-1', 'fov-2-scan-1']
        fovs_extracted = io_utils.list_folders(extraction_dir)
        assert fovs.sort() == fovs_extracted.sort()

        # when given only moly fovs will raise error
        moly_bin_file_dir = os.path.join(Path(__file__).parent, "data", "moly")
        with pytest.raises(FileNotFoundError, match="No viable bin files were found"):
            bin_extraction.extract_missing_fovs(moly_bin_file_dir, extraction_dir,
                                                panel, extract_intensities=False)
