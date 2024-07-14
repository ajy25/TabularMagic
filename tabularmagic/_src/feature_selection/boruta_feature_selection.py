from .base_feature_selection import BaseFSC


class BorutaSelector(BaseFSC):
    def __init__(self, name: str | None = None):
        """
        Constructs a BorutaSelector.

        Parameters
        ----------
        - name : str | None.

        Returns
        -------
        - None
        """
        super().__init__(name)
