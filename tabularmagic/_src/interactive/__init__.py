from .eda import ComprehensiveEDA
from .regression.voteselectreg import RegressionVotingSelectionReport
from .regression.mlreg import MLRegressionReport
from .regression.linreg import LinearRegressionReport
from .regression.glmreg import GLMRegressionReport
from .classification.mlclass import MLClassificationReport


__all__ = [
    "ComprehensiveEDA",
    "RegressionVotingSelectionReport",
    "MLRegressionReport",
    "LinearRegressionReport",
    "GLMRegressionReport",
    "MLClassificationReport",
]
