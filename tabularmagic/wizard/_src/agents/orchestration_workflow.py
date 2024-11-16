from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
import pandas as pd

from json import dumps


from ..options import options
from ..tools.tooling_context import ToolingContext
from ..io.vector_store import VectorStoreManager
from ..io.datacontainer import DataContainer
from ..io.canvas import CanvasQueue

from ..._src import build_tabularmagic_analyzer


from .eda_agent import build_eda_agent
from .linear_regression_agent import build_linear_regression_agent
from .ml_agent import build_ml_agent


class VarDescriptionSetupEvent(Event):
    var_to_desc: dict


class EDAAgentEvent(Event):
    info: str


class MLAgentEvent(Event):
    info: str


class LinearRegressionAgentEvent(Event):
    info: str


class WizardFlow(Workflow):

    def __init__(
        self,
        df: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        test_size: float = 0.2,
        timeout: int = 60,
        verbose: bool = False,
    ):
        super().__init__(
            timeout=timeout,
            verbose=verbose,
        )

        self._llm: FunctionCallingLLM = options.llm_build_function()
        data_container = DataContainer()
        data_container.set_analyzer(
            build_tabularmagic_analyzer(df, df_test=df_test, test_size=test_size)
        )
        self._context = ToolingContext(
            data_container=data_container,
            vectorstore_manager=VectorStoreManager(),
            canvas_queue=CanvasQueue(),
        )

        self._eda_agent = build_eda_agent(
            llm=self._llm, context=self._context, react=False
        )
        self._ml_agent = build_ml_agent(
            llm=self._llm, context=self._context, react=False
        )
        self._linear_regression_agent = build_linear_regression_agent(
            llm=self._llm, context=self._context, react=False
        )

    @step
    async def setup_var_descriptions(self, ev: StartEvent) -> VarDescriptionSetupEvent:
        """Asks the LLM to guess the variable descriptions."""

        prompt = "We will analyze a dataset. Here are the variables of the dataset: "
        prompt += ", ".join(self._context.data_container.df.columns) + ". "
        prompt += "Guess the descriptions of these variables. "
        prompt += "Respond in the format: 'variable_name: description', "
        prompt += "separated by commas between variables. "
        prompt += "Do not include any other text in your response."
        response = await self._llm.acomplete(prompt)
        response = str(response)

        var_to_desc = {}
        for pair in response.split(","):
            var, desc = pair.split(":")
            var_to_desc[var.strip()] = desc.strip()

        self._context.add_dict(var_to_desc)

        return VarDescriptionSetupEvent(var_to_desc=var_to_desc)

    @step
    async def finish(self, ev: VarDescriptionSetupEvent) -> StopEvent:
        return StopEvent(result=dumps(ev.var_to_desc))
