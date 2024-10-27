import logging
from pathlib import Path

debug_path = Path(__file__).resolve().parent
debug_log_path = debug_path / "_debug_log.log"

WIZARD_LOGGER = logging.Logger("Wizard Debug Log", level=logging.DEBUG)
handler = logging.FileHandler(filename=debug_log_path)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
WIZARD_LOGGER.addHandler(handler)


def print_debug(message: str):
    WIZARD_LOGGER.debug(message)


with open(debug_log_path, "w") as f:
    f.write("")
