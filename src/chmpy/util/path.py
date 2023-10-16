import errno
import os
import sys
from pathlib import Path


ERR_INVALID_NAME = 123  # windows specific error code


def list_directory(pathname):
    return "\n".join(
        f"{str(p):<60s} {p.lstat().st_size:>20d}B" for p in Path(pathname).iterdir()
    )


def is_valid_pathname(pathname):
    """Return `True` if the passed string is a valid
    pathname for the current OS, `False` otherwise"""
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip the drivename on windows
        _, pathname = os.path.splitdrive(pathname)

        root_dirname = (
            os.environ.get("HOMEDRIVE", "C:")
            if sys.platform == "win32"
            else os.path.sep
        )
        assert os.path.isdir(root_dirname)

        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            except OSError as e:
                if hasattr(e, "winerror"):
                    if e.winerror == ERR_INVALID_NAME:
                        return False
                elif e.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False

    except TypeError:
        return False

    return True


def is_path_creatable(pathname):
    """Return `True` if we have permissions to create
    the given pathname, `False` otherwise"""
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)


def path_exists_or_is_creatable(pathname):
    """Return `True` if the given pathname is valid for
    the current OS and currently exists or is createable
    by the current user, `False` otherwise."""
    try:
        return is_valid_pathname(pathname) and (
            os.path.exists(pathname) or is_path_creatable(pathname)
        )
    except OSError:
        return False


def dir_exists_or_is_creatable(pathname):
    """Return `True` if the given pathname is valid for
    the current OS and currently exists as a dir or is
    createable by the current user, `False` otherwise."""
    try:
        if not is_valid_pathname(pathname):
            return False
        return (
            os.path.exists(pathname) and os.path.isdir(pathname)
        ) or is_path_creatable(pathname)
    except OSError:
        return False


def dir_exists(pathname):
    """Return `True` if the given pathname is valid for
    the current OS and currently exists as a dir,
    `False` otherwise."""
    try:
        if not is_valid_pathname(pathname):
            return False
        return os.path.exists(pathname) and os.path.isdir(pathname)
    except OSError:
        return False
