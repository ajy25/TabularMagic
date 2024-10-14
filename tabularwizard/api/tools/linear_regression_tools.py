from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field

from ..tabularmagic_utils import GLOBAL_DATA_CONTAINER
from ..io.jsonutils import save_dict_to_json, save_df_to_json


# OLS regression tool
class OLSRegressionTool(BaseModel):
    formula: str = Field(
        description="The formula for the OLS regression model. "
        "Use the tilde (~) symbol to separate the dependent and independent variables. "
        "For example, 'y ~ x1 + x2'."
    )


def ols_regression_function(formula: str) -> str:
    """Performs an OLS regression analysis using the specified formula."""
    lm_report = GLOBAL_DATA_CONTAINER.analyzer.ols(formula=formula)
    return save_dict_to_json(lm_report._to_dict())


ols_regression_tool = FunctionTool.from_defaults(
    fn=ols_regression_function,
    name="ols_regression_tool",
    description="Performs an OLS regression analysis using the specified formula. "
    "The formula should be in the form 'y ~ x1 + x2', where 'y' is the dependent "
    "variable and 'x1', 'x2', etc. are the independent variables. "
    "Returns a JSON string containing the regression results.",
    fn_schema=OLSRegressionTool,
)
