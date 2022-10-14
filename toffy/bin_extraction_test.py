import os
import tempfile
import pandas as pd
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, call
import natsort as ns

from toffy import bin_extraction
from ark.utils import test_utils
from mibi_bin_tools import io_utils, bin_files


@patch('builtins.print')
def test_extract_missing_fovs(mocked_print):
    bin_file_dir = os.path.join(Path(__file__).parent, "data", "combined")

    panel = pd.DataFrame([{
        'Mass': 98,
        'Target': None,
        'Start': 97.5,
        'Stop': 98.5,
    }])

    # only 1 fov that has empty json should raise a warning
    with tempfile.TemporaryDirectory() as tmpdir:
        test_utils._make_blank_file(tmpdir, 'fov-1-scan-1.bin')
        test_utils._make_blank_file(tmpdir, 'fov-1-scan-1.json')

        with pytest.warns(Warning) as warninfo:
            bin_extraction.extract_missing_fovs(tmpdir, tmpdir, panel, extract_intensities=False)
        warns = {(warn.category, warn.message.args[0]) for warn in warninfo}
        assert warns == {(UserWarning, "The following FOVs have empty json files and will not be "
                                       "processed:\n ['fov-1-scan-1']"),
                         (Warning, f"No viable bin files were found in {tmpdir}")}

    # test that it does not re-extract fovs
    with tempfile.TemporaryDirectory() as extraction_dir:
        os.makedirs(os.path.join(extraction_dir, 'fov-1-scan-1'))
        bin_extraction.extract_missing_fovs(bin_file_dir, extraction_dir,
                                            panel, extract_intensities=False)

        assert mocked_print.mock_calls == \
               [call('Skipping the following previously extracted FOVs: ', 'fov-1-scan-1'),
                call('Moly FOVs which will not be extracted: ', 'fov-3-scan-1'),
                call('Found 2 FOVs to extract.'),
                call('Extraction completed!')]
    mocked_print.reset_mock()

    # test that it does not re-extract fovs and no moly extraction
    with tempfile.TemporaryDirectory() as combined_bin_file_dir:

        for file_name in io_utils.list_files(bin_file_dir):
            shutil.copy(os.path.join(bin_file_dir, file_name),
                        os.path.join(combined_bin_file_dir, file_name))

        # create empty json fov
        test_utils._make_blank_file(combined_bin_file_dir, 'empty.bin')
        test_utils._make_blank_file(combined_bin_file_dir, 'empty.json')

        # check for correct output
        with tempfile.TemporaryDirectory() as extraction_dir:
            os.makedirs(os.path.join(extraction_dir, 'fov-1-scan-1'))
            with pytest.warns(UserWarning, match='empty json files'):
                bin_extraction.extract_missing_fovs(combined_bin_file_dir, extraction_dir,
                                                    panel, extract_intensities=False)

            assert mocked_print.mock_calls == \
                   [call('Skipping the following previously extracted FOVs: ', 'fov-1-scan-1'),
                    call('Moly FOVs which will not be extracted: ', 'fov-3-scan-1'),
                    call('FOVs with empty json files which will not be extracted: ', 'empty'),
                    call('Found 2 FOVs to extract.'),
                    call('Extraction completed!')]

            # when given empty fov files will raise a warning
            with pytest.warns(Warning, match="The following FOVs have empty json files"):
                bin_extraction.extract_missing_fovs(combined_bin_file_dir, extraction_dir,
                                                    panel, extract_intensities=False)

            # test that neither moly nor empty fov were extracted
            assert ns.natsorted(io_utils.list_folders(extraction_dir)) \
                   == ns.natsorted(['fov-2-scan-1', 'fov-1-scan-2', 'fov-1-scan-1'])

    # test successful extraction of fovs
    with tempfile.TemporaryDirectory() as extraction_dir:
        mocked_print.reset_mock()
        bin_extraction.extract_missing_fovs(bin_file_dir, extraction_dir,
                                            panel, extract_intensities=False)
        fovs = ['fov-1-scan-1', 'fov-2-scan-1']
        fovs_extracted = io_utils.list_folders(extraction_dir)

        # test no extra print statements
        assert mocked_print.mock_calls == \
               [call('Moly FOVs which will not be extracted: ', 'fov-3-scan-1'),
                call('Found 3 FOVs to extract.'), call('Extraction completed!')]

        # check both fovs were extracted
        assert fovs.sort() == fovs_extracted.sort()

        # all fovs extracted except moly already will raise a warning
        with pytest.warns(Warning, match="No viable bin files were found"):
            bin_extraction.extract_missing_fovs(bin_file_dir, extraction_dir,
                                                panel, extract_intensities=False)
