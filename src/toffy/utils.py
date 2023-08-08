import os
import pathlib
import stat
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from shutil import _OnErrorCallback


def remove_readonly(func: Callable, path: pathlib.Path | str, excinfo: "_OnErrorCallback"):
    """
    Removes readonly files, mainly useful for CI/CD pipelines.
    Reference: https://stackoverflow.com/questions/1889597/deleting-read-only-directory-in-python

    Example usage:
    shutil.rmtree(my_path, onerror=remove_readonly)

    Args:
        func (Callable): The function to be called, e.g. `shutil.rmtree`.
        path (pathlib.Path | str): The path to the file / directory.
        excinfo (shutil._OnErrorCallback): The exception callabck.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)
