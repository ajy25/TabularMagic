from ..util.console import color_text
from ..util.constants import TOSTR_MAX_WIDTH, TOSTR_ROUNDING_N_DECIMALS
from textwrap import fill


class StatisticalTestResult:
    """StatisticalTestResult. Class for presenting statistical testing 
    results.
    """

    def __init__(self, 
                 description: str, 
                 statistic: float, 
                 pval: float,
                 descriptive_statistic: float = None, 
                 degfree: float = None,
                 statistic_description: str = None, 
                 descriptive_statistic_description: str = None,
                 null_hypothesis_description: str = None, 
                 alternative_hypothesis_description: str = None, 
                 long_description: str = None):
        """
        Parameters
        ----------
        - description: str.
        - statistic : float. The statistic of the test. For example, the 
            t-statistic for the two-sample t-test.
        - pval : float. 
        - descriptive_statistic : float. The statistic that describes the
            values tested. For example, Pearson correlation coefficient for 
            correlation test, or difference in means for two-sample t-test.
        - degfree : float. Degrees of freedom. 
        - statistic_description : str.
        - descriptive_statistic_description : str.
        - null_hypothesis_description : str.
        - alternative_hypothesis_description : str.
        - long_description : str.
        
        """
        self._description = description
        self._descriptive_statistic = descriptive_statistic
        self._statistic = statistic
        self._pval = pval
        self._degfree = degfree
        self._descriptive_statistic_description =\
              descriptive_statistic_description
        self._statistic_description = statistic_description
        self._null_hypothesis_description = null_hypothesis_description
        self._alternative_hypothesis_description =\
              alternative_hypothesis_description
        self._long_description = long_description
        

    
    def pval(self):
        return self._pval
    
    def statistic(self):
        return self._statistic



    def __str__(self):
        """Returns data and metadata in string form."""
        
        max_width = TOSTR_MAX_WIDTH
        n_dec = TOSTR_ROUNDING_N_DECIMALS

        top_divider = color_text('='*max_width, 'none') + '\n'
        bottom_divider = '\n' + color_text('='*max_width, 'none')
        divider = '\n' + color_text('-'*max_width, 'none') + '\n'
        divider_invisible = '\n' + ' '*max_width + '\n'
       

        description_message = fill(f'{self._description}', max_width)
        pval_message = fill(f'p-value: {round(self._pval, n_dec)}', max_width)
        statistic_message = fill(
            f'{self._statistic_description}: {round(self._statistic, n_dec)}', 
            max_width
        )


        supplementary_message = divider[:-1]
        if self._null_hypothesis_description:
            supplementary_message += '\n'
            supplementary_message += fill(
                f'H0: {self._null_hypothesis_description}', max_width)
        if self._alternative_hypothesis_description:
            supplementary_message += '\n'
            supplementary_message += fill(
                f'HA: {self._alternative_hypothesis_description}', max_width)
        if self._descriptive_statistic and\
               self._descriptive_statistic_description:
            supplementary_message += '\n'
            supplementary_message += fill(
                f'{self._descriptive_statistic_description}: ' +\
                f'{round(self._descriptive_statistic, n_dec)}', max_width)
        if self._degfree:
            supplementary_message += '\n'
            supplementary_message += fill(
                'Degrees of freedom: ' +\
                f'{round(self._degfree, n_dec)}', max_width)
        if self._long_description:
            supplementary_message += divider
            supplementary_message += self._long_description


        final_message = top_divider + description_message +\
            divider + statistic_message + '\n' + pval_message +\
            supplementary_message + bottom_divider


        return final_message


    def _repr_pretty_(self, p, cycle):
        p.text(str(self))









    







