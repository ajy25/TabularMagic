import numpy as np
import json
from ..display.print_utils import color_text, bold_text, fill_ignore_format
from ..display.print_options import print_options


class StatisticalTestResult:
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
        assumptions_description: str | None = None,
        long_description: str | None = None,
    ):
        """
        Parameters
        ----------
        description: str.
        statistic : float.
            The statistic of the test. For example, the t-statistic for the
            two-sample t-test.
        pval : float.
        descriptive_statistic : float.
            Default: None. The statistic that describes the values tested.
            For example, Pearson correlation coefficient for correlation test,
            or difference in means for two-sample t-test.
        degfree : float.
            Default: None. Degrees of freedom.
        statistic_description : str.
            Default: None. Description of the statistic.
        descriptive_statistic_description : str.
            Default: None. Description of the descriptive statistic.
        null_hypothesis_description : str.
            Default: None. Description of the null hypothesis.
        alternative_hypothesis_description : str.
            Default: None. Description of the alternative hypothesis.
        assumptions_description : str.
            Default: None. Description of the assumptions of the test.
        long_description : str.
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
        self._assumptions_description = assumptions_description
        self._long_description = long_description

    def pval(self):
        """Returns the p-value."""
        return self._pval

    def statistic(self):
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

        max_width = print_options.max_line_width
        n_dec = print_options.n_decimals

        top_divider = color_text("=" * max_width, "none") + "\n"
        bottom_divider = "\n" + color_text("=" * max_width, "none")
        divider = "\n" + color_text("-" * max_width, "none") + "\n"
        # divider_invisible = '\n' + ' '*max_width + '\n'

        description_message = fill_ignore_format(
            bold_text(self._description), max_width
        )

        pval_str = f"{self._pval:.{n_dec}e}"
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
            + color_text(pval_str, "yellow")
            + " " * pval_str_message_buffer_right,
            max_width,
        )

        supplementary_message = divider[:-1]
        if self._null_hypothesis_description is not None:
            supplementary_message += "\n"
            supplementary_message += fill_ignore_format(
                bold_text("H0: ")
                + color_text(self._null_hypothesis_description, "yellow"),
                max_width,
            )
        if self._alternative_hypothesis_description is not None:
            supplementary_message += "\n"
            supplementary_message += fill_ignore_format(
                bold_text("HA: ")
                + color_text(self._alternative_hypothesis_description, "yellow"),
                max_width,
            )
        if (
            self._descriptive_statistic is not None
            and self._descriptive_statistic_description is not None
        ):
            supplementary_message += "\n"
            supplementary_message += fill_ignore_format(
                bold_text(f"{self._descriptive_statistic_description}: ")
                + color_text(str(round(self._descriptive_statistic, n_dec)), "yellow"),
                max_width,
            )
        if self._degfree is not None:
            supplementary_message += "\n"
            supplementary_message += fill_ignore_format(
                bold_text("Degrees of freedom: ")
                + color_text(str(round(self._degfree, n_dec)), "yellow"),
                max_width,
            )
        if self._assumptions_description is not None:
            supplementary_message += "\n"
            supplementary_message += fill_ignore_format(
                bold_text("Assumptions: ")
                + color_text(self._assumptions_description, "yellow"),
                max_width,
                subsequent_indent=13,
            )
        if self._long_description is not None:
            supplementary_message += divider
            supplementary_message += fill_ignore_format(
                bold_text("More info: ") + color_text(self._long_description, "yellow"),
                subsequent_indent=11,
            )

        final_message = (
            top_divider
            + description_message
            + divider
            + statistic_pval_message
            + supplementary_message
            + bottom_divider
        )

        return final_message

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
