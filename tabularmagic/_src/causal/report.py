import numpy as np
from ..display.print_options import print_options
from ..display.print_utils import (
    color_text,
    bold_text,
    list_to_string,
    fill_ignore_format,
    format_two_column,
)


class CausalReport:
    """Class for storing and displaying causal inference results."""

    def __init__(
        self,
        estimate: float,
        se: float,
        n_units: int,
        outcome_var: str,
        treatment_var: str,
        confounders: list[str],
        estimand: str,
        method: str,
        method_description: str,
    ):
        """Initializes a CausalReport object.

        Parameters
        ----------
        estimate : float
            The estimate of the causal effect.

        se : float
            The standard error of the estimator.

        n_units : int
            The number of units in the data.

        outcome_var : str
            The name of the outcome variable.

        treatment_var : str
            The name of the treatment variable.

        confounders : list[str]
            The names of the confounding variables.

        estimand : str
            The estimand of the causal effect. Either "ate" or "att".

        method : str
            The method used to estimate the causal effect.

        method_description : str
            A description of the method used to estimate the causal effect.
        """

        self._estimate = estimate
        self._estimate_se = se
        self._n_units = n_units
        self._outcome_var = outcome_var
        self._treatment_var = treatment_var
        self._confounders = confounders
        self._estimand = estimand
        self._method = method
        self._method_description = method_description

    def __str__(self):
        max_width = print_options._max_line_width
        n_dec = print_options._n_decimals

        top_divider = color_text("=" * max_width, "none") + "\n"
        bottom_divider = "\n" + color_text("=" * max_width, "none")
        divider = "\n" + color_text("-" * max_width, "none") + "\n"
        divider_invisible = "\n" + " " * max_width + "\n"

        title_message = bold_text("Causal Effect Estimation Report")

        estimand = (
            "Avg Tmt Effect" if self._estimand == "ate" else "Avg Tmt Effect on Treated"
        )
        estimate_message = ""
        estimate_message += format_two_column(
            f"{bold_text('Estimated ' + estimand + ':')} "
            f"{color_text(f'{self._estimate:.{n_dec}f}', 'yellow')}",
            f"{bold_text('Std Err:')} "
            f"{color_text(f'{self._estimate_se:.{n_dec}f}', 'yellow')}",
            max_width,
        )

        treatment_message = bold_text("Treatment variable:\n") + color_text(
            "  '" + self._treatment_var + "'", "purple"
        )

        outcome_message = bold_text("Outcome variable:\n") + color_text(
            "  '" + self._outcome_var + "'", "purple"
        )

        confounders_message = bold_text("Confounders:\n") + fill_ignore_format(
            list_to_string(self._confounders), initial_indent=2, subsequent_indent=2
        )

        return (
            top_divider
            + title_message
            + divider
            + estimate_message
            + divider
            + treatment_message
            + divider_invisible
            + outcome_message
            + divider_invisible
            + confounders_message
            + bottom_divider
        )

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(self.__class__.__name__ + "(...)")
        else:
            p.text(str(self))
