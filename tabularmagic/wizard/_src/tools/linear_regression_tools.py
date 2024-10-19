from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field

from json import dumps

from ..datacontainer import GLOBAL_DATA_CONTAINER
from ..io.global_io import GLOBAL_IO


# OLS tool
class _OLSTool(BaseModel):
    formula: str = Field(
        description="Formula for the least squares regression. " 
        "For example, y ~ x1 + x2. "
        "The formula should be in the form of a string. "
    )


def _ols_function(formula: str) -> str:
    """Performs ordinary least squares regression."""
    ols_report = GLOBAL_DATA_CONTAINER.analyzer.ols(formula=formula.strip())
    output_str = GLOBAL_IO.add_str(dumps(ols_report._to_dict()))
    output_str += "\n" + GLOBAL_IO.add_figure(
        ols_report.plot_diagnostics("train"), text_description="Diagnostic plots."
    )
    return output_str


ols_tool = FunctionTool.from_defaults(
    fn=_ols_function,
    name="ols_tool",
    description="Performs ordinary least squares regression. "
    "Returns a JSON string containing coefficients and metrics. "
    "Also, plots diagnostic plots. "
    "Detailed text describing the diagnostic plots will be saved to STORAGE, "
    "as well as returned. "
    "The output string will be added to STORAGE.",
    fn_schema=_OLSTool,
)
