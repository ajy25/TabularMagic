import logging
from pathlib import Path

debug_path = Path(__file__).resolve().parent
logger_path = debug_path / "_log.txt"

WIZARD_LOGGER = logging.Logger("tabularwizard", level=logging.DEBUG)
handler = logging.FileHandler(filename=logger_path)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
WIZARD_LOGGER.addHandler(handler)


def print_debug(message: str):
    WIZARD_LOGGER.debug(message)


with open(logger_path, "w") as f:
    f.write("")
