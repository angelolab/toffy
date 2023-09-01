import errno
import os
import pathlib
import shutil
import stat
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from shutil import _OnErrorCallback


def remove_readonly(func: Callable, path: pathlib.Path | str, excinfo: "_OnErrorCallback"):
    """
    Removes readonly files, mainly useful for Windows CI/CD pipelines.
    References:
        - https://stackoverflow.com/questions/1889597/deleting-read-only-directory-in-python
        - https://stackoverflow.com/questions/2656322/shutil-rmtree-fails-on-windows-with-access-is-denied

    Example usage:
    shutil.rmtree(my_path, onerror=remove_readonly)

    Args:
        func (Callable): The function to be called, e.g. `shutil.rmtree`.
        path (pathlib.Path | str): The path to the file / directory.
        excinfo (shutil._OnErrorCallback): The exception callabck.
    """
    # os.chmod(path, stat.S_IWRITE)
    excvalue = excinfo[1]
    if func in (os.rmdir, os.remove, shutil.rmtree) and excvalue.errno == errno.EACCES:
        if not os.access(path, os.W_OK):
            os.chmod(
                path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO | stat.S_IWRITE | stat.S_IWUSR
            )  # 0777
            func(path)
    else:
        raise
