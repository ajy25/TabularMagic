from pathlib import Path


path_to_scratch = Path(__file__).parent / "_scratch"
path_to_scratch.mkdir(exist_ok=True)


class _ScratchTracker:
    """Tracks the scratchpad. Prevents overflow of the scratchpad by deleting
    the oldest scratch.
    """

    def __init__(self):
        """Initializes the scratch tracker."""
        self._filepath = path_to_scratch / "scratch.txt"
        self.clear_scratch()

    def write_scratch(self, scratch: str) -> str:
        """Writes a scratch to the scratchpad and updates the scratch tracker.
        Returns the scratch.

        Parameters
        ----------
        scratch : str
            The scratch to write.

        Returns
        -------
        str
            The scratch.
        """
        with open(self._filepath, "a") as f:  # Open in append mode
            f.write("\n\n" + scratch)
        return scratch

    def read_scratch(self) -> str:
        """Reads a scratch from the scratchpad and returns it.

        Returns
        -------
        str
            The scratch.
        """
        with open(self._filepath, "r") as f:
            scratch = f.read()
        return scratch

    def clear_scratch(self) -> None:
        """Clears the scratchpad."""
        with open(self._filepath, "w") as f:
            f.write("")


scratch_tracker = _ScratchTracker()
