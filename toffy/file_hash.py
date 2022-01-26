# adapted from https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
import hashlib
import os
from ark.utils import io_utils, misc_utils

import warnings


def get_hash(filepath):
    """Computes the hash of the specified file to verify file integrity

    Args:
        filepath (str | PathLike): full path to file

    Returns:
        string: the hash of the file"""

    with open(filepath, "rb") as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def compare_directories(dir_1, dir_2):
    """Compares two directories to ensure all files are present in both with the same hashes

    Args:
        dir_1: first directory to compare
        dir_2: second directory to compare

    Returns:
        list: a list of files with different hashes between the two directories"""

    dir_1_folders = io_utils.list_folders(dir_1)
    dir_2_folders = io_utils.list_folders(dir_2)

    if len(dir_1_folders) > 0:
        warnings.warn("The following subfolders were found in the first directory. Sub-folder "
                      "contents will not be compared for accuracy, if you want to ensure "
                      "successful copying please run this function on those subdirectories. "
                      "{}".format(dir_1_folders))

    if len(dir_2_folders) > 0:
        warnings.warn("The following subfolders were found in the second directory. Sub-folder "
                      "contents will not be compared for accuracy, if you want to ensure "
                      "successful copying please run this function on those subdirectories. "
                      "{}".format(dir_2_folders))

    dir_1_files = io_utils.list_files(dir_1)
    dir_2_files = io_utils.list_files(dir_2)

    misc_utils.verify_same_elements(directory_1=dir_1_files, directory_2=dir_2_files)

    bad_files = []
    for file in dir_1_files:
        hash1 = get_hash(os.path.join(dir_1, file))
        hash2 = get_hash(os.path.join(dir_2, file))

        if hash1 != hash2:
            print("Found a file with differing hashes: {}".format(file))
            bad_files.append(file)

    return bad_files
