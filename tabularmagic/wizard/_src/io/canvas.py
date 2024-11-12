from pathlib import Path


class CanvasQueue:
    """Class for storing tables and figures in order.
    Works only with Path objects.
    First in, first out.
    """

    def __init__(self):
        """Initializes the CanvasQueue object."""
        self._fig_queue = []
        self._table_queue = []
        self._all_figs = []
        self._all_tables = []

    def push_figure(self, path: Path):
        """Adds a figure to the queue.

        Parameters
        ----------
        path : Path
            The path to the figure.
        """
        self._fig_queue.append(path)
        self._all_figs.append(path)

    def push_table(self, path: Path):
        """Adds a table to the queue.

        Parameters
        ----------
        path : Path
            The path to the table.
        """
        self._table_queue.append(path)
        self._all_tables.append(path)

    def pop_figure(self) -> Path:
        return self._fig_queue.pop(0)

    def pop_table(self) -> Path:
        return self._table_queue.pop(0)
