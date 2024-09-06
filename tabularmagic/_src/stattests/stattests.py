import numpy as np
import json
from ..display.print_utils import color_text, bold_text, fill_ignore_format
from ..display.print_options import print_options


class StatisticalTestReport:
    """Class for storing and displaying statistical test results."""

    def __init__(
        self,
        description: str,
        statistic: float,
        pval: float,
        descriptive_statistic: float | None = None,
        degfree: float | None = None,
        statistic_description: str | None = None,
        descriptive_statistic_description: str | None = None,
        null_hypothesis_description: str | None = None,
        alternative_hypothesis_description: str | None = None,
        assumptions_description: str | list | None = None,
        long_description: str | None = None,
    ):
        """Initializes a StatisticalTestResult object.

        Parameters
        ----------
        description : str
            A description of the statistical test.

        statistic : float
            The statistic of the test. For example, the t-statistic for the
            two-sample t-test.

        pval : float
            The p-value of the test.

        descriptive_statistic : float | None
            Default: None. The statistic that describes the values tested.
            For example, Pearson correlation coefficient for correlation test,
            or difference in means for two-sample t-test.

        degfree : float | None
            Default: None. Degrees of freedom.

        statistic_description : str | None
            Default: None. Description of the statistic.

        descriptive_statistic_description : str | None
            Default: None. Description of the descriptive statistic.

        null_hypothesis_description : str | None
            Default: None. Description of the null hypothesis.

        alternative_hypothesis_description : str | None
            Default: None. Description of the alternative hypothesis.

        assumptions_description : str | list | None
            Default: None. Description of the assumptions of the test.

        long_description : str
            Default: None. A long description of the test.
        """

        self._description = description
        self._descriptive_statistic = descriptive_statistic
        self._statistic = statistic
        self._pval = pval
        self._degfree = degfree
        self._descriptive_statistic_description = descriptive_statistic_description
        self._statistic_description = statistic_description
        self._null_hypothesis_description = null_hypothesis_description
        self._alternative_hypothesis_description = alternative_hypothesis_description

        if isinstance(assumptions_description, list):
            self._assumptions_description = ""
            for i, assumption in enumerate(assumptions_description):
                self._assumptions_description += f"{i+1}. {assumption}"
                if i != len(assumptions_description) - 1:
                    self._assumptions_description += "\n"
        else:
            self._assumptions_description = assumptions_description

        self._long_description = long_description

    def pval(self) -> float:
        """Returns the p-value."""
        return self._pval

    def statistic(self) -> float:
        """Returns the statistic."""
        return self._statistic

    def _agentic_describe_json_str(self) -> str:
        """Returns a JSON string for the agentic agent."""
        return json.dumps(
            {
                "description": self._description,
                "statistic": self._statistic,
                "p_value": self._pval,
                "descriptive_statistic": self._descriptive_statistic,
                "degrees_of_freedom": self._degfree,
                "statistic_description": self._statistic_description,
                "descriptive_statistic_description": self._descriptive_statistic_description,
                "null_hypothesis_description": self._null_hypothesis_description,
                "alternative_hypothesis_description": self._alternative_hypothesis_description,
                "assumptions_description": self._assumptions_description,
                "long_description": self._long_description,
            }
        )

    def __str__(self):
        """Returns data and metadata in string form."""

        max_width = print_options._max_line_width
        n_dec = print_options._n_decimals

        top_divider = color_text("=" * max_width, "none") + "\n"
        bottom_divider = "\n" + color_text("=" * max_width, "none")
        divider = "\n" + color_text("-" * max_width, "none") + "\n"

        description_message = fill_ignore_format(
            bold_text(self._description), max_width
        )

        pval_str = f"{self._pval:.{n_dec}e}"

        if self._pval <= 0.05:
            pval_color = "green"
        else:
            pval_color = "red"

        statistic_str = str(round(self._statistic, n_dec))
        textlen_pval = (
            len(pval_str)
            + len(statistic_str)
            + len("p-value: ")
            + len(self._statistic_description + ": ")
        )
        pval_str_message_buffer_left = (max_width - textlen_pval) // 2
        pval_str_message_buffer_right = int(np.ceil((max_width - textlen_pval) / 2))
        statistic_pval_message = fill_ignore_format(
            bold_text(self._statistic_description + ": ")
            + color_text(statistic_str, "yellow")
            + " " * pval_str_message_buffer_left
            + bold_text("p-value: ")
            + color_text(pval_str, pval_color)
            + " " * pval_str_message_buffer_right,
            max_width,
        )

        supplementary_message = ""

        if self._null_hypothesis_description is not None:
            supplementary_message += "\n\n"
            supplementary_message += bold_text("H0:") + "\n"
            supplementary_message += fill_ignore_format(
                color_text(self._null_hypothesis_description, "blue"),
                max_width,
                initial_indent=2,
                subsequent_indent=2,
            )
        if self._alternative_hypothesis_description is not None:
            supplementary_message += "\n\n"
            supplementary_message += bold_text("HA:") + "\n"
            supplementary_message += fill_ignore_format(
                color_text(self._alternative_hypothesis_description, "blue"),
                max_width,
                initial_indent=2,
                subsequent_indent=2,
            )
        if (
            self._descriptive_statistic is not None
            and self._descriptive_statistic_description is not None
        ):
            supplementary_message += "\n\n"
            supplementary_message += (
                fill_ignore_format(
                    bold_text(f"{self._descriptive_statistic_description}:"), max_width
                )
                + "\n"
            )
            supplementary_message += fill_ignore_format(
                color_text(str(round(self._descriptive_statistic, n_dec)), "yellow"),
                max_width,
                initial_indent=2,
                subsequent_indent=2,
            )
        if self._degfree is not None:
            supplementary_message += "\n\n"
            supplementary_message += (
                fill_ignore_format(bold_text("Degrees of freedom:"), max_width) + "\n"
            )
            supplementary_message += fill_ignore_format(
                color_text(str(round(self._degfree, n_dec)), "yellow"),
                max_width,
                initial_indent=2,
                subsequent_indent=2,
            )
        if self._assumptions_description is not None:
            supplementary_message += "\n\n"
            supplementary_message += (
                fill_ignore_format(bold_text("Assumptions:"), max_width) + "\n"
            )
            supplementary_message += fill_ignore_format(
                color_text(self._assumptions_description, "blue"),
                max_width,
                initial_indent=2,
                subsequent_indent=5,
            )
        if self._long_description is not None:
            supplementary_message += divider
            supplementary_message += (
                fill_ignore_format(bold_text("More info:"), max_width) + "\n"
            )
            supplementary_message += fill_ignore_format(
                color_text(self._long_description, "blue"),
                max_width,
                initial_indent=2,
                subsequent_indent=2,
            )

        if len(supplementary_message) > 0:
            if supplementary_message[0:2] == "\n\n":
                supplementary_message = supplementary_message[2:]

        final_message = (
            top_divider
            + description_message
            + divider
            + statistic_pval_message
            + divider
            + supplementary_message
            + bottom_divider
        )

        return final_message

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
