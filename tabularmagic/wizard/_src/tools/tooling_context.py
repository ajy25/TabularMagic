import matplotlib.pyplot as plt
import pandas as pd
from json import dumps
from ..._src import DataContainer, VectorStoreManager, CanvasQueue


class ToolingContext:

    def __init__(
        self,
        data_container: DataContainer,
        vectorstore_manager: VectorStoreManager,
        canvas_queue: CanvasQueue,
    ):
        """Initializes the ToolingContext object.

        Parameters
        ----------
        data_container : DataContainer
            The DataContainer object to use for storing data.

        vectorstore_manager : VectorStoreManager
            The VectorStoreManager object to use for storing data.
        """
        self._data_container = data_container
        self._vectorstore_manager = vectorstore_manager
        self._canvas_queue = canvas_queue

    def add_figure(
        self,
        fig: plt.Figure,
        text_description: str,
        augment_text_description: bool = True,
    ) -> str:
        """Adds a figure.

        Parameters
        ----------
        fig : plt.Figure
            Figure to add to the vector index.

        text_description : str
            Description of the figure.

        augment_text_description : bool
            Whether to augment the text description with a vision model,
            by default True

        Returns
        -------
        str
            Description of the figure.
        """
        descr, path = self._vectorstore_manager.add_figure(
            fig=fig,
            text_description=text_description,
            augment_text_description=augment_text_description,
        )
        self._canvas_queue.push_figure(path)
        return descr

    def add_str(self, text: str) -> str:
        """Adds a string.

        Parameters
        ----------
        text : str
            Text to add to the vector index.

        Returns
        -------
        str
            The input text, verbatim.
        """
        return self._vectorstore_manager.add_str(text)

    def add_table(self, table: pd.DataFrame, add_to_vectorstore: bool = True) -> str:
        """Adds a pandas DataFrame.

        Parameters
        ----------
        table : dict
            Table to add to the canvas (and optionally, the vector index).

        add_to_vectorstore : bool
            Whether to add the table to the vector index, by default True.
            May be set to False if a custom dict including the DataFrame
            is to be added to the vector store (e.g. use add_dict instead).

        Returns
        -------
        str
            The input table in json string format.
        """
        strres, path = self._vectorstore_manager.add_table(table, add_to_vectorstore)
        self._canvas_queue.push_table(path)
        return strres

    def add_dict(self, dictionary: dict) -> str:
        """Adds a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary to add to the vector index.

        Returns
        -------
        str
            The input dictionary in json string format.
        """
        str_dict = dumps(dictionary)
        return self._vectorstore_manager.add_str(str_dict)

    @property
    def data_container(self) -> DataContainer:
        return self._data_container

    @property
    def vectorstore_manager(self) -> VectorStoreManager:
        return self._vectorstore_manager

    @property
    def canvas_queue(self) -> CanvasQueue:
        return self._canvas_queue
