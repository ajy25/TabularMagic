from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
import pandas as pd

from json import dumps


from ..options import options
from ..tools.tooling_context import ToolingContext
from ..io.storage_manager import StorageManager
from ..io.datacontainer import DataContainer
from ..io.canvas import CanvasQueue

from .. import build_tabularmagic_analyzer


from .eda_agent import build_eda_agent
from .linear_regression_agent import build_linear_regression_agent
from .ml_agent import build_ml_agent

from .._debug.logger import print_debug


class VarDescriptionSetupEvent(Event):
    pass


class VarDescriptionSetupErrorEvent(Event):
    objective: str
    error: str
    number: int


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
        self._tooling_context = ToolingContext(
            data_container=data_container,
            storage_manager=StorageManager(),
            canvas_queue=CanvasQueue(),
        )

        self._eda_agent = build_eda_agent(
            llm=self._llm, context=self._tooling_context, react=False
        )
        self._ml_agent = build_ml_agent(
            llm=self._llm, context=self._tooling_context, react=False
        )
        self._linear_regression_agent = build_linear_regression_agent(
            llm=self._llm, context=self._tooling_context, react=False
        )

    @step
    async def setup_var_descriptions(
        self, ctx: Context, ev: StartEvent | VarDescriptionSetupErrorEvent
    ) -> VarDescriptionSetupEvent | VarDescriptionSetupErrorEvent:
        """Asks the LLM to guess the variable descriptions."""

        objective = ev.get("objective")
        if objective is None:
            raise ValueError("Objective is required.")
        await ctx.set("objective", objective)
        prompt = """We will analyze a dataset. Here are the variables of the dataset: 
        {variables}. 
        Guess the descriptions of these variables. 
        Respond in the format: 'variable_name: description', separated by commas between variables. 
        Do not include any other text in your response.
        """.format(
            variables=", ".join(self._tooling_context.data_container.df.columns)
        )

        if isinstance(ev, VarDescriptionSetupErrorEvent):
            number = ev.number
            prompt += (
                "\n\nPreviously, this error occurred in your response: " + ev.error
            )
        else:
            number = 0

        if number > 3:
            raise RuntimeError("Too many attempts. Please try again later.")

        response = await self._llm.acomplete(prompt)
        response = str(response)

        var_to_desc = {}
        try:
            for pair in response.split(","):
                var, desc = pair.split(":")
                var_to_desc[var.strip()] = desc.strip()
        except Exception as e:
            return VarDescriptionSetupErrorEvent(
                objective=objective,
                error=f"Invalid response format: {e}. Please try again.",
                number=number + 1,
            )

        self._tooling_context.add_dict(var_to_desc)

        print_debug(
            "(setup_var_descriptions) Variable to description dictionary: {var_to_desc}".format(
                var_to_desc=dumps(var_to_desc)
            )
        )

        await ctx.set("var_to_description_dict", var_to_desc)

        return VarDescriptionSetupEvent()

    @step
    async def run_eda(
        self,
        ctx: Context,
        ev: VarDescriptionSetupEvent,
    ) -> EDAAgentEvent:
        """Runs the EDA agent."""
        objective = await ctx.get("objective")
        var_to_description_dict = await ctx.get("var_to_description_dict")

        prompt = """
        We will perform exploratory data analysis on the dataset.

        Here is the overarching objective: {objective}.

        Here are the variables and their descriptions:
        {var_to_description_dict}.

        Given this overarching objective and the variables, formulate a list of relevant analyses to run.
        
        Here are the options:
        1. Compute summary statistics for each variable.
        2. Compare the means of a variable across different groups.
        3. Describe the distribution of a variable.
        4. Compare the correlation of a variable with other variables.
        5. Compute the correlation matrix.

        Respond with instructions, separated by semicolons. You MUST use the proper variable names.
        For example (assuming a dataset with numeric variables 'MPG', 'Horsepower', 'Weight' and categorical variable 'Cylinders'):
        "Compute summary statistics; Compare the means of 'MPG' between 'Cylinders' groups; Describe the distribution of 'MPG'; Find the correlation between 'MPG' and the following: 'Horsepower, Weight'".
        """.format(
            objective=objective, var_to_description_dict=dumps(var_to_description_dict)
        )

        print_debug("(run_eda) LLM Prompt: {prompt}".format(prompt=prompt))

        response = await self._llm.acomplete(prompt)
        response = str(response)

        list_of_instructions = response.split(";")
        responses = []
        for instruction in list_of_instructions:
            print_debug(
                "(run_eda) EDA Agent instruction: {instruction}".format(
                    instruction=instruction
                )
            )
            agent_response = self._eda_agent.chat(instruction)
            responses.append(str(agent_response))

        result = "\n\n\n".join(responses)

        print_debug("(run_eda) EDA Agent response: {result}".format(result=result))

        return EDAAgentEvent(info=result)

    @step
    async def finish(self, ev: EDAAgentEvent) -> StopEvent:
        info = ev.info
        print_debug(f"(finish) EDA Agent result: {info}")
        return StopEvent(result=info)
