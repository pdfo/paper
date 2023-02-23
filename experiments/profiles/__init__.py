import functools
import logging

from profiles.profiles import Profiles

_log = logging.getLogger(__name__)


@functools.lru_cache()
def _ensure_handler():
    """
    Attach a console handler to the root logger.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s"))
    _log.addHandler(handler)
    return handler


def set_loglevel(level):
    """
    Set the root logger and the root logger basic handler levels to `level`.
    """
    _log.setLevel(level)
    _ensure_handler().setLevel(level)


__all__ = ["Profiles", "set_loglevel"]
