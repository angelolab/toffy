import os

from ark.utils import io_utils
from json_utils import list_moly_fovs
from mibi_bin_tools import bin_files


def extract_missing_fovs(bin_file_dir, extraction_dir, panel, extract_intensities):

    # retrieve all fov names from base_dir and extracted fovs from extraction_dir
    fovs = io_utils.remove_file_extensions(io_utils.list_files(bin_file_dir, substrs='.bin'))
    extracted_fovs = io_utils.list_folders(extraction_dir, substrs='fov')

    # filter out moly fovs
    moly_fovs = list_moly_fovs(bin_file_dir)

    print("Previous extracted FOVs: ", extracted_fovs)
    print("Moly FOVs which will not be extracted: ", moly_fovs)

    # extract missing fovs to extraction_dir
    non_moly_fovs = list(set(fovs).difference(moly_fovs))
    missing_fovs = list(set(non_moly_fovs).difference(extracted_fovs))
    bin_files.extract_bin_files(bin_file_dir, extraction_dir, include_fovs=missing_fovs,
                                panel=panel, intensities=extract_intensities)
