from .._src.ml.discriminative.regression import (
    LinearR,
    RobustLinearR,
    TreeR,
    TreeEnsembleR,
    MLPR,
    SVMR,
    CustomR,
)
from .._src.interactive.regression.mlreg import MLRegressionReport


from .._src.interactive.classification.mlclass import MLClassificationReport
from .._src.ml.discriminative.classification import (
    LinearC,
    TreeC,
    TreeEnsembleC,
    MLPC,
    SVMC,
    CustomC,
)


__all__ = [
    "LinearR",
    "RobustLinearR",
    "TreeR",
    "TreeEnsembleR",
    "MLPR",
    "SVMR",
    "CustomR",
    "MLRegressionReport",
    "MLClassificationReport",
    "LinearC",
    "TreeC",
    "TreeEnsembleC",
    "CustomC",
    "MLPC",
    "SVMC",
]
