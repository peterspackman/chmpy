import os
import platform


def is_executable(path):
    return os.path.isfile(path) and os.access(path, os.X_OK)


def which(prog):
    fpath, fname = os.path.split(prog)
    if fpath and is_executable(fname):
        return prog
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, prog)
            if is_executable(exe_file):
                return exe_file
    return None


def linux_version():
    """Get the version of linux this system is running
    If it's not linux, return None
    """
    s = platform.platform()
    if not s.startswith("Linux"):
        return None
    version_string = s.split("-")[1]
    return version_string


def libc_version():
    """Get the version of glibc this python was compiled with
    return None if we don't have any info on it
    """
    try:
        version = platform.libc_ver()[1]
    except Exception:
        version = None
    return version
