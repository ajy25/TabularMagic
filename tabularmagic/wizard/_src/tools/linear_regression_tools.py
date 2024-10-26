from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from json import dumps
from functools import partial
from .tooling_context import ToolingContext


# OLS tool
class _OLSToolInput(BaseModel):
    formula: str = Field(
        description="Formula for the least squares regression. "
        "For example, y ~ x1 + x2. "
        "The formula should be in the form of a string. "
    )


def _ols_function(formula: str, context: ToolingContext) -> str:
    """Performs ordinary least squares regression."""
    ols_report = context._data_container.analyzer.ols(formula=formula.strip())
    output_str = context._vectorstore_manager.add_str(dumps(ols_report._to_dict()))
    output_str += "\n" + context._vectorstore_manager.add_figure(
        ols_report.plot_diagnostics("train"), text_description="Diagnostic plots."
    )
    return output_str


def build_ols_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_ols_function, context=context),
        name="ols_tool",
        description="Performs ordinary least squares regression. "
        "Returns a JSON string containing coefficients and metrics. "
        "Also, plots diagnostic plots. "
        "Detailed text describing the diagnostic plots will be saved to STORAGE, "
        "as well as returned. "
        "The output string will be added to STORAGE.",
        fn_schema=_OLSToolInput,
    )
