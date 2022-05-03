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


@parametrize_with_cases('callbacks, kwargs', cases=ExtractionQCGenerationCases)
@parametrize_with_cases('data_path',  cases=ExtractionQCCallCases)
def test_build_fov_callback(callbacks, kwargs, data_path):

    intensities = kwargs.get('intensities', False)

    with tempfile.TemporaryDirectory() as tmp_dir:

        extracted_dir = os.path.join(tmp_dir, 'extracted')
        qc_dir = os.path.join(tmp_dir, 'qc')
        kwargs['tiff_out_dir'] = extracted_dir
        kwargs['qc_out_dir'] = qc_dir

        # test cb generates w/o errors
        cb = watcher_callbacks.build_fov_callback(*callbacks, **kwargs)

        point_names = io_utils.list_files(data_path, substrs=['bin'])
        point_names = [name.split('.')[0] for name in point_names]

        for name in point_names:
            cb(data_path, name)

        # just check SMA
        if 'extract_tiffs' in callbacks:
            check_extraction_dir_structure(extracted_dir, point_names, ['SMA'], intensities)
        if 'genereate_qc' in callbacks:
            check_qc_dir_structure(qc_dir, point_names)
