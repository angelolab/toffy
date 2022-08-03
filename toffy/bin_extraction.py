import warnings

import natsort as ns

from toffy.json_utils import list_moly_fovs, check_for_empty_files
from mibi_bin_tools import bin_files, io_utils


def extract_missing_fovs(bin_file_dir, extraction_dir, panel, extract_intensities, replace=True):
    """Check for already extracted FOV bin files, and extract the remaining
    (excluding moly fovs and fovs with empty json files)

    Args:
        bin_file_dir (str): path to directory containing the bin and json files
        extraction_dir (str): path to directory of already extracted FOVs
        panel (pd.DataFrame): file defining the panel info for bin file extraction
        extract_intensities (bool): whether to extract intensities from the bin files
        replace (bool): whether to replace pulse images with intensity
    """

    # retrieve all fov names from base_dir and extracted fovs from extraction_dir
    fovs = io_utils.remove_file_extensions(io_utils.list_files(bin_file_dir, substrs='.bin'))
    extracted_fovs = io_utils.list_folders(extraction_dir, substrs='fov')

    # filter out empty json file fovs
    empty_fovs = check_for_empty_files(bin_file_dir)
    if empty_fovs:
        fovs = list(set(fovs).difference(empty_fovs))

    # check for moly fovs
    moly_fovs = list_moly_fovs(bin_file_dir, fovs)

    if extracted_fovs:
        print("Skipping the following previously extracted FOVs: ", ", ".join(extracted_fovs))
    if moly_fovs:
        print("Moly FOVs which will not be extracted: ", ", ".join(moly_fovs))
    if empty_fovs:
        print("FOVs with empty json files which will not be extracted: ", ", ".join(empty_fovs))

    # extract missing fovs to extraction_dir
    non_moly_fovs = list(set(fovs).difference(moly_fovs))
    missing_fovs = list(set(non_moly_fovs).difference(extracted_fovs))
    missing_fovs = ns.natsorted(missing_fovs)

    if missing_fovs:
        print(f"Found {len(missing_fovs)} FOVs to extract.")
        bin_files.extract_bin_files(bin_file_dir, extraction_dir, include_fovs=missing_fovs,
                                    panel=panel, intensities=extract_intensities, replace=replace)
    else:
        warnings.warn(f"No viable bin files were found in {bin_file_dir}", UserWarning)
