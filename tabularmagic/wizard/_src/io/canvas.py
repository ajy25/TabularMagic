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
        self._all_analysis = [] # combine tables and figures
        self._analysis_queue = []

    def push_figure(self, path: Path):
        """Adds a figure to the queue.

        Parameters
        ----------
        path : Path
            The path to the figure.
        """
        self._fig_queue.append(path)
        self._all_figs.append(path)
        self._analysis_queue.append(path)
        self._all_analysis.append(path)

    def push_table(self, path: Path):
        """Adds a table to the queue.

        Parameters
        ----------
        path : Path
            The path to the table.
        """
        self._table_queue.append(path)
        self._all_tables.append(path)
        self._analysis_queue.append(path)
        self._all_analysis.append(path)

    def pop_figure(self) -> Path:
        fig_path = self._fig_queue.pop(0)
        if fig_path in self._analysis_queue:
            self._analysis_queue.remove(fig_path)
        else:
            raise ValueError("Path not found in the analysis queue.")
        return fig_path

    def pop_table(self) -> Path:
        table_path = self._table_queue.pop(0)
        if table_path in self._analysis_queue:
            self._analysis_queue.remove(table_path)
        else:
            raise ValueError("Path not found in the analysis queue.")
        return table_path
    
    def pop_analysis(self) -> Path:
        path = self._analysis_queue.pop(0)
        if path in self._table_queue:
            self._table_queue.remove(path)
        elif path in self._fig_queue:
            self._fig_queue.remove(path)
        else:
            raise ValueError("Path not found in table or figure queue.")
        return path
    
    def get_figures(self) -> list[Path]:
        return self._all_figs
    
    def get_tables(self) -> list[Path]:
        return self._all_tables
    
    def get_analysis(self) -> list[Path]:
        return self._all_analysis


    