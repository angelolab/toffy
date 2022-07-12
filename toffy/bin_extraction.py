from toffy.json_utils import list_moly_fovs
from mibi_bin_tools import bin_files, io_utils


def extract_missing_fovs(bin_file_dir, extraction_dir, panel, extract_intensities, replace=True):
    """Check for already extracted FOV bin files, and extract the remaining (excluding moly)

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

    # filter out moly fovs
    moly_fovs = list_moly_fovs(bin_file_dir)

    print("Skipping the following previously extracted FOVs: ", ", ".join(extracted_fovs))
    print("Moly FOVs which will not be extracted: ", ", ".join(moly_fovs))

    # extract missing fovs to extraction_dir
    non_moly_fovs = list(set(fovs).difference(moly_fovs))
    missing_fovs = list(set(non_moly_fovs).difference(extracted_fovs))

    if missing_fovs:
        bin_files.extract_bin_files(bin_file_dir, extraction_dir, include_fovs=missing_fovs,
                                    panel=panel, intensities=extract_intensities, replace=replace)
    else:
        raise Warning("No viable bin files were found in ", bin_file_dir)
