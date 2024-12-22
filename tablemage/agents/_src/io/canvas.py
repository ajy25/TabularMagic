from pathlib import Path


class CanvasItem:
    pass


class CanvasTable(CanvasItem):
    def __init__(self, path: Path):
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def __str__(self):
        return str(self._path)


class CanvasFigure(CanvasItem):
    def __init__(self, path: Path):
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def __str__(self):
        return str(self._path)


class CanvasThought(CanvasItem):
    def __init__(self, thought: str):
        self._thought = thought

    @property
    def thought(self) -> str:
        return self._thought

    def __str__(self):
        return self._thought


class CanvasCode(CanvasItem):
    def __init__(self, code: str):
        self._code = code

    @property
    def code(self) -> str:
        return self._code

    def __str__(self):
        return self._code


class CanvasQueue:
    """Class for storing tables and figures in order.
    Works only with Path objects.
    First in, first out.
    """

    def __init__(self):
        """Initializes the CanvasQueue object."""
        self._fig_queue: list[CanvasFigure] = []
        self._all_figs: list[CanvasFigure] = []

        self._table_queue: list[CanvasTable] = []
        self._all_tables: list[CanvasTable] = []

        self._code_queue: list[CanvasCode] = []
        self._all_code: list[CanvasCode] = []

        self._thought_queue: list[CanvasThought] = []
        self._all_thoughts: list[CanvasThought] = []

        # combine tables, figures, code, and thoughts
        self._analysis_queue: list[CanvasItem] = []
        self._all_analysis: list[CanvasItem] = []

    def push_figure(self, path: Path):
        """Adds a figure to the queue.

        Parameters
        ----------
        path : Path
            The path to the figure.
        """
        fig = CanvasFigure(path)
        self._fig_queue.append(fig)
        self._all_figs.append(fig)
        self._analysis_queue.append(fig)
        self._all_analysis.append(fig)

    def push_table(self, path: Path):
        """Adds a table to the queue.

        Parameters
        ----------
        path : Path
            The path to the table.
        """
        table = CanvasTable(path)
        self._table_queue.append(table)
        self._all_tables.append(table)
        self._analysis_queue.append(table)
        self._all_analysis.append(table)

    def push_code(self, code: str):
        """Adds code to the queue.

        Parameters
        ----------
        code : str
            The code to add.
        """
        code = CanvasCode(code)
        self._code_queue.append(code)
        self._all_code.append(code)
        self._analysis_queue.append(code)
        self._all_analysis.append(code)

    def push_thought(self, thought: str):
        """Adds a thought to the queue.

        Parameters
        ----------
        thought : str
            The thought to add.
        """
        thought = CanvasThought(thought)
        self._thought_queue.append(thought)
        self._all_thoughts.append(thought)
        self._analysis_queue.append(thought)
        self._all_analysis.append(thought)

    def pop_figure(self) -> CanvasFigure:
        fig = self._fig_queue.pop(0)
        if fig in self._analysis_queue:
            self._analysis_queue.remove(fig)
        else:
            raise ValueError("Path not found in the analysis queue.")
        return fig

    def pop_table(self) -> CanvasTable:
        table = self._table_queue.pop(0)
        if table in self._analysis_queue:
            self._analysis_queue.remove(table)
        else:
            raise ValueError("Path not found in the analysis queue.")
        return table

    def pop_code(self) -> CanvasCode:
        code = self._code_queue.pop(0)
        if code in self._analysis_queue:
            self._analysis_queue.remove(code)
        else:
            raise ValueError("Path not found in the analysis queue.")
        return code

    def pop_thought(self) -> CanvasThought:
        thought = self._thought_queue.pop(0)
        if thought in self._analysis_queue:
            self._analysis_queue.remove(thought)
        else:
            raise ValueError("Path not found in the analysis queue.")
        return thought

    def pop_analysis(self) -> CanvasItem:
        item = self._analysis_queue.pop(0)
        if item in self._table_queue:
            self._table_queue.remove(item)
        elif item in self._fig_queue:
            self._fig_queue.remove(item)
        else:
            raise ValueError("Path not found in table or figure queue.")
        return item

    def get_figures(self) -> list[CanvasFigure]:
        return self._all_figs

    def get_tables(self) -> list[CanvasTable]:
        return self._all_tables

    def get_code(self) -> list[CanvasCode]:
        return self._all_code

    def get_thoughts(self) -> list[CanvasThought]:
        return self._all_thoughts

    def get_analysis(self) -> list[CanvasItem]:
        return self._all_analysis
