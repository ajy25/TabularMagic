from .base_feature_selection import BaseFeatureSelector


class BorutaFeatureSelector(BaseFeatureSelector):

    def __init__(self, name: str | None = None):
        """
        Constructs a BorutaFeatureSelector.

        Parameters
        ----------
        - name : str | None.

        Returns
        -------
        - None
        """
        super().__init__(name)
