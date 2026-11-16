"""
Internal helpers for warnings and logging.
"""
import logging
import warnings

logger = logging.getLogger("bicm")
logger.addHandler(logging.NullHandler())


class BiCMWarning(UserWarning):
    """Warning emitted by the bicm package.

    Silence it with ``warnings.filterwarnings("ignore", category=bicm.BiCMWarning)``.
    """


def warn(message, category=BiCMWarning, stacklevel=3):
    """Emit a warning pointing at the caller of the public bicm method."""
    warnings.warn(message, category, stacklevel=stacklevel)
