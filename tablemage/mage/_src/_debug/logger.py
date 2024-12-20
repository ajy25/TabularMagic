import logging
from pathlib import Path
import sys


debug_path = Path(__file__).resolve().parent
debug_log_path = debug_path / "_debug_log.log"
if not debug_log_path.exists():
    debug_log_path.touch()

MAGE_LOGGER = logging.Logger("Mage Log", level=logging.DEBUG)
handler = logging.FileHandler(filename=debug_log_path)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
MAGE_LOGGER.addHandler(handler)

# let's add sys.stderr to the logger
stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setLevel(logging.DEBUG)
MAGE_LOGGER.addHandler(stream_handler)


def print_debug(message: str):
    MAGE_LOGGER.debug(message)


with open(debug_log_path, "w") as f:
    f.write("")
