"""
Internal helpers for warnings, logging and multiprocessing.
"""
import logging
import multiprocessing
import warnings
from platform import system

logger = logging.getLogger("bicm")
logger.addHandler(logging.NullHandler())


class BiCMWarning(UserWarning):
    """Warning emitted by the bicm package.

    Silence it with ``warnings.filterwarnings("ignore", category=bicm.BiCMWarning)``.
    """


def warn(message, category=BiCMWarning, stacklevel=3):
    """Emit a warning pointing at the caller of the public bicm method."""
    warnings.warn(message, category, stacklevel=stacklevel)


def get_pool(processes):
    """Return a process pool.

    On Linux the "fork" start method is used, as Python did by default up to 3.13.
    Since Python 3.14 the default is "forkserver", which re-imports the user's main
    script and fails in scripts without an ``if __name__ == "__main__":`` guard.
    Other systems keep their default start method.
    """
    if system() == "Linux":
        context = multiprocessing.get_context("fork")
    else:
        context = multiprocessing.get_context()
    return context.Pool(processes=processes)
