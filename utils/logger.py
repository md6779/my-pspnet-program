from __future__ import annotations

import logging
import sys

from termcolor import colored


def create_logger(name: str = "") -> logging.Logger:
    """Create logger.

    Returns:
        logging.Logger: Logger.
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # color fomatter
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "cyan")
        + colored("(%(filename)s %(lineno)d)", "yellow")  # noqa
        + ": %(levelname)s | %(message)s"  # noqa
    )

    # create console handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)
    return logger
