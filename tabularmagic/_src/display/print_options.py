import logging
import pathlib
import sys


def make_default_logger() -> logging.Logger:
    logger = logging.Logger(name="Default TabularMagic Logger")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger


class _PrintOptions:
    """Class for setting and tracking options for printing and logging."""

    def __init__(self):
        """Initializes the a _ConsoleOptions object with default settings."""
        self._logger = make_default_logger()
        self._muted = False

        self.n_decimals = 3
        self.max_line_width = 80

    def _log_info(self, msg: str):
        if not self._muted:
            self._logger.info(msg)

    def _log_debug(self, msg: str):
        if not self._muted:
            self._logger.debug(msg)

    def reset_logger(self, logger: logging.Logger = None):
        """Sets a new logger.

        Parameters
        ----------
        logger : logging.Logger.
            Default : None. If None, resets the logger to the default.
        """
        if logger is None:
            self._logger = make_default_logger()
        else:
            self._logger = logger

    def add_log_file(self, path: pathlib.Path | str, level: int = logging.DEBUG):
        """Adds a file handler to the logger.

        Parameters
        ----------
        path : pathlib.Path | str.
            Path to the .log or .txt file.
        level : int.
            Default: logging.DEBUG.
        """
        file_handler = logging.FileHandler(str(path))
        file_handler.setLevel(level)
        self._logger.addHandler(file_handler)

    def mute(self):
        self._muted = True

    def unmute(self):
        self._muted = False


print_options = _PrintOptions()
