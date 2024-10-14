import logging
from pathlib import Path

debug_path = Path(__file__).resolve().parent

WIZARD_LOGGER = logging.Logger(
    "tabularwizard",
    level=logging.DEBUG
)
WIZARD_LOGGER.addHandler(logging.FileHandler(filename=debug_path / "_log.txt"))
    
def print_debug(message: str):
    WIZARD_LOGGER.debug(message)
    

