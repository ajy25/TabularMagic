from ..._src import DataContainer, VectorStoreManager


class ToolingContext:

    def __init__(
        self,
        data_container: DataContainer,
        vectorstore_manager: VectorStoreManager,
    ):
        """Initializes the ToolingContext object.

        Parameters
        ----------
        data_container : DataContainer
            The DataContainer object to use for storing data.

        vectorstore_manager : VectorStoreManager
            The VectorStoreManager object to use for storing data.
        """
        self._data_container = data_container  # Use the property setter
        self._vectorstore_manager = vectorstore_manager  # Use the property setter

    @property
    def data_container(self) -> DataContainer:
        return self._data_container

    @property
    def vectorstore_manager(self) -> VectorStoreManager:
        return self._vectorstore_manager
