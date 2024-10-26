import pandas as pd
from .._src import (
    build_tabularmagic_analyzer,
    WizardIO,
    DataContainer,
    print_debug,
)
from ..._src.display.print_utils import suppress_all_output, suppress_logging
from .._src.agents.orchestrator_agent import OrchestratorAgent
from .._src.llms.openai.openai import build_openai
from .._src.tools.tooling_context import ToolingContext


class Wizard:
    """Class for interacting with the LLMs for data analysis on tabular data."""

    def __init__(
        self,
        df: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        test_size: float = 0.2,
    ):
        """Initializes the Wizard object.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to build the Analyzer for.

        df_test : pd.DataFrame | None
            The test DataFrame to use for the Analyzer. Defaults to None.

        test_size : float
            The size of the test set. Defaults to 0.2.
        """

        self.data_container = DataContainer()
        self.data_container.set_analyzer(
            build_tabularmagic_analyzer(df, df_test=df_test, test_size=test_size)
        )
        print_debug(
            "Data container initialized with the Analyzer built from the "
            "provided DataFrame."
        )
        self.io = WizardIO()
        self.context = ToolingContext(
            data_container=self.data_container,
            wizard_io=self.io,
        )
        print_debug("IO initialized.")

        self._orchestrator_agent = OrchestratorAgent(
            llm=build_openai(), context=self.context
        )

    def chat(self, message: str) -> str:
        """Interacts with the LLM to provide data analysis insights.

        Parameters
        ----------
        message : str
            The message to send to the LLM.

        Returns
        -------
        str
            The response from the LLM.
        """
        # with suppress_all_output(), suppress_logging():
        return self._orchestrator_agent.chat(message)
