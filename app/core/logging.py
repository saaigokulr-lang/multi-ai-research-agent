"""Centralized logging configuration.

We configure the standard library ``logging`` module once at import time so
every module in the app shares one formatter and level, instead of each
module (or worse, ``print()``) inventing its own output format.
"""

import logging
import sys

from app.core.config import get_settings

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_root_logger() -> None:
    """Attach a single stream handler with our formatter to the root logger.

    Guarded by checking for existing handlers (rather than a module-level
    flag) so repeated ``get_logger`` calls don't stack duplicate handlers.
    """
    root = logging.getLogger()
    if root.handlers:
        return

    settings = get_settings()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))

    root.setLevel(settings.LOG_LEVEL)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, ensuring root configuration has run."""
    _configure_root_logger()
    return logging.getLogger(name)
