import logging
import sys

debug_logger = logging.Logger("tabularwizard")
debug_logger.setLevel(logging.DEBUG)
stdoutstream = logging.StreamHandler(sys.stdout)
stdoutstream.setLevel(logging.DEBUG)
debug_logger.addHandler(stdoutstream)


def printl(message: str, level: int = logging.DEBUG):
    """Logs a message. Level is set to DEBUG.
    
    Parameters
    ----------
    message : str
        The message to log.

    level : int, optional
        The logging level, by default logging.DEBUG.
    """
    debug_logger.log(level, "TabularWizard :: " + message)
    

