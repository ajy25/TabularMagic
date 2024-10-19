from ..._src import DataContainer, WizardIO


class ToolingContext:

    def __init__(
        self,
        data_container: DataContainer,
        wizard_io: WizardIO,
    ):
        """Initializes the ToolingContext object.

        Parameters
        ----------
        data_container : DataContainer
            The DataContainer object to use for storing data.

        wizard_io : WizardIO
            The WizardIO object to use for storing data.
        """
        self.data_container = data_container  # Use the property setter
        self.io = wizard_io  # Use the property setter

    @property
    def data_container(self) -> DataContainer:
        return self._data_container

    @data_container.setter
    def data_container(self, data_container: DataContainer):
        self._data_container = data_container

    @property
    def io(self) -> WizardIO:
        return self._io

    @io.setter
    def io(self, io: WizardIO):
        self._io = io
