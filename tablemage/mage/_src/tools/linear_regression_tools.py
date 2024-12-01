from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from functools import partial
from .tooling_context import ToolingContext
from .._debug.logger import print_debug


# OLS tool
class _OLSToolInput(BaseModel):
    formula: str = Field(
        description="""Formula for the least squares regression.
        Target variable must be numeric.
        For example, y ~ x1 + x2.
        The formula should be a string."""
    )


def _ols_function(formula: str, context: ToolingContext) -> str:
    """Performs ordinary least squares regression."""
    print_debug(f"_ols_function call: " f"formula: {formula}")
    ols_report = context._data_container.analyzer.ols(formula=formula.strip())
    print_debug(f"summary: {str(ols_report)}")
    context.add_table(ols_report.metrics("both"), add_to_vectorstore=False)
    context.add_table(ols_report.coefs(), add_to_vectorstore=False)
    output_str = context.add_dict(ols_report._to_dict())
    output_str += "\n" + context.add_figure(
        ols_report.plot_diagnostics("train"), text_description="Diagnostic plots."
    )
    print_debug(f"_ols_function output: {output_str}")
    return output_str


def build_ols_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_ols_function, context=context),
        name="ols_tool",
        description="""Performs ordinary least squares regression.
        Returns a JSON string containing coefficients and metrics.
        Also, plots diagnostic plots.""",
        fn_schema=_OLSToolInput,
    )


# Logit tool
class _LogitToolInput(BaseModel):
    formula: str = Field(
        description="""Formula for the logistic regression.
        Target variable must be categorical.
        For example, y ~ x1 + x2.
        The formula should be a string."""
    )


def _logit_function(formula: str, context: ToolingContext) -> str:
    """Performs logistic regression."""
    print_debug(f"_logit_function call: " f"formula: {formula}")
    logit_report = context._data_container.analyzer.logit(formula=formula.strip())
    context.add_table(logit_report.metrics("both"), add_to_vectorstore=False)
    context.add_table(logit_report.coefs(), add_to_vectorstore=False)
    output_str = context.add_dict(logit_report._to_dict())
    output_str += "\n" + context.add_figure(
        logit_report.plot_diagnostics("train"), text_description="Diagnostic plots."
    )
    print_debug(f"_logit_function output: {output_str}")
    return output_str


def build_logit_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_logit_function, context=context),
        name="logit_tool",
        description="""Performs logistic regression.
        Returns a JSON string containing coefficients and metrics.
        Also, plots diagnostic plots.""",
        fn_schema=_LogitToolInput,
    )
