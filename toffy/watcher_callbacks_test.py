import os
import tempfile
from pytest_cases import parametrize_with_cases

from mibi_bin_tools import io_utils

from toffy import watcher_callbacks
from toffy.test_utils import (
    ExtractionQCGenerationCases,
    ExtractionQCCallCases,
    check_extraction_dir_structure,
    check_qc_dir_structure,
)


@parametrize_with_cases('panel,  kwargs', cases=ExtractionQCGenerationCases,
                        has_tag='extract')
@parametrize_with_cases('data_path',  cases=ExtractionQCCallCases)
def test_build_extraction_callback(panel, kwargs, data_path):

    intensities = kwargs.get('intensities', False)

    with tempfile.TemporaryDirectory() as tmp_dir:

        extracted_dir = os.path.join(tmp_dir, 'extracted')
        os.makedirs(extracted_dir)

        # test cb generates w/o errors
        cb = watcher_callbacks.build_extract_callback(extracted_dir, panel, **kwargs)

        point_names = io_utils.list_files(data_path, substrs=['bin'])
        point_names = [name.split('.')[0] for name in point_names]

        for name in point_names:
            cb(data_path, name)

        # just check SMA
        check_extraction_dir_structure(extracted_dir, point_names, ['SMA'], intensities)


@parametrize_with_cases('panel,  kwargs', cases=ExtractionQCGenerationCases,
                        has_tag='qc')
@parametrize_with_cases('data_path',  cases=ExtractionQCCallCases)
def test_build_qc_callback(panel, kwargs, data_path):

    with tempfile.TemporaryDirectory() as tmp_dir:

        qc_dir = os.path.join(tmp_dir, 'qc')
        os.makedirs(qc_dir)

        # test cb generates w/o errors
        cb = watcher_callbacks.build_qc_callback(qc_dir, panel, **kwargs)

        point_names = io_utils.list_files(data_path, substrs=['bin'])
        point_names = [name.split('.')[0] for name in point_names]

        for name in point_names:
            cb(data_path, name)

        check_qc_dir_structure(qc_dir, point_names)
