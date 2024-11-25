import numpy as np
from typing import Literal
from sklearn.metrics import f1_score, roc_curve


def select_optimal_threshold_binary(
    y_true: np.ndarray,
    y_pred_score: np.ndarray,
    metric: Literal["f1", "roc", None] = "f1",
    threshold_range: tuple = (0.01, 0.99),
    n_thresholds: int = 200,
) -> float:
    """Selects the optimal threshold for binary classification.

    Parameters
    ----------
    y_true : np.ndarray ~ (sample_size,)
        True labels (0 or 1).
    y_pred_score : np.ndarray ~ (sample_size,)
        Predicted probability scores for the positive class.
    metric : str, optional (default="f1")
        Metric to use for threshold optimization:
        - "f1": F1 score
        - "roc": ROC curve (maximizes sensitivity + specificity)
        - None: No threshold optimization
    threshold_range : tuple, optional (default=(0.01, 0.99))
        The range of thresholds to search over (min, max).
    n_thresholds : int, optional (default=200)
        Number of thresholds to evaluate.

    Returns
    -------
    float | None
        Optimal threshold value. None if metric is None.
    """
    if metric is None:
        return None

    if metric == "f1":
        # Optimize using F1 score
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        f1_scores = []

        for threshold in thresholds:
            y_pred = (y_pred_score > threshold).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx]

    elif metric == "roc":
        # Optimize using ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_score)

        # Find threshold that maximizes sensitivity + specificity
        optimal_idx = np.argmax(tpr - fpr)

        # If roc_thresholds is empty or out of range, use default
        if len(roc_thresholds) > 0:
            optimal_threshold = roc_thresholds[optimal_idx]
            # Ensure threshold is within specified range
            optimal_threshold = np.clip(
                optimal_threshold, threshold_range[0], threshold_range[1]
            )
            return optimal_threshold

        return 0.5

    else:
        raise ValueError(f"Invalid metric: {metric}")


def predict_with_threshold_binary(
    y_pred_score: np.ndarray, threshold: float | None
) -> np.ndarray:
    """Apply optimal threshold to make binary predictions.

    Parameters
    ----------
    y_pred_score : np.ndarray ~ (sample_size,)
        Predicted probability scores for the positive class.
    threshold : float
        Classification threshold.
        If None, uses 0.5 as the default threshold.

    Returns
    -------
    np.ndarray ~ (sample_size,)
        Binary predictions (0 or 1).
    """
    if threshold is None:
        threshold = 0.5
    return (y_pred_score > threshold).astype(int)


def select_optimal_thresholds_multiclass(
    y_true: np.ndarray,
    y_pred_score: np.ndarray,
    metric: Literal["f1", "roc", None] = "f1",
    threshold_range: tuple = (0.1, 0.9),
    n_thresholds: int = 100,
) -> list[list[float]]:
    """Selects the optimal thresholds for multiclass classification using
    one-vs-one approach.

    Parameters
    ----------
    y_true : np.ndarray ~ (sample_size,)
        True labels.
    y_pred_score : np.ndarray ~ (sample_size, n_classes)
        Predicted probability scores for each class.
    metric : str, optional (default="f1")
        Metric to use for threshold optimization:
        - "f1": F1 score
        - "roc": ROC curve (maximizes sensitivity + specificity)
    threshold_range : tuple, optional (default=(0.1, 0.9))
        The range of thresholds to search over (min, max).
    n_thresholds : int, optional (default=100)
        Number of thresholds to evaluate.

    Returns
    -------
    List[List[float]] | None
        Matrix of optimal thresholds for each class pair.
        None if metric is None.
    """
    if metric is None:
        return None

    n_classes = y_pred_score.shape[1]
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
    optimal_thresholds = [[0.5 for _ in range(n_classes)] for _ in range(n_classes)]

    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            # Get samples belonging to classes i and j
            mask = (y_true == i) | (y_true == j)
            if not np.any(mask):
                continue

            # Extract relevant samples and convert to binary problem
            y_true_binary = (y_true[mask] == i).astype(int)
            y_pred_binary = y_pred_score[mask][:, [i, j]]

            # Convert to probability ratios for class i vs j
            prob_ratios = y_pred_binary[:, 0] / (
                y_pred_binary[:, 0] + y_pred_binary[:, 1]
            )

            if metric == "f1":
                # Optimize using F1 score
                f1_scores = []
                for threshold in thresholds:
                    y_pred = (prob_ratios > threshold).astype(int)
                    f1_scores.append(f1_score(y_true_binary, y_pred, zero_division=0))
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx]

            elif metric == "roc":
                # Optimize using ROC curve
                fpr, tpr, roc_thresholds = roc_curve(y_true_binary, prob_ratios)
                # Find threshold that maximizes sensitivity + specificity
                optimal_idx = np.argmax(tpr - fpr)
                # If roc_thresholds is empty or out of range, use default
                if (
                    len(roc_thresholds) > 0
                    and threshold_range[0]
                    <= roc_thresholds[optimal_idx]
                    <= threshold_range[1]
                ):
                    optimal_threshold = roc_thresholds[optimal_idx]
                else:
                    optimal_threshold = 0.5

            else:
                raise ValueError(f"Invalid metric: {metric}")

            # Store optimal thresholds symmetrically
            optimal_thresholds[i][j] = optimal_threshold
            optimal_thresholds[j][i] = 1 - optimal_threshold

    return optimal_thresholds


def predict_with_thresholds_multiclass(
    y_pred_score: np.ndarray, thresholds: list[list[float]] | None = None
) -> np.ndarray:
    """Apply optimal thresholds to make predictions using one-vs-one voting.

    Parameters
    ----------
    y_pred_score : np.ndarray ~ (sample_size, n_classes)
        Predicted probability scores for each class.
    thresholds : List[List[float]] | None
        Matrix of optimal thresholds for each class pair.
        If None, obtain predictions via argmax.

    Returns
    -------
    np.ndarray ~ (sample_size,)
        Predicted class labels.
    """
    if thresholds is None:
        return np.argmax(y_pred_score, axis=1)

    n_samples, n_classes = y_pred_score.shape
    votes = np.zeros((n_samples, n_classes))

    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            # Add small epsilon to prevent division by zero
            epsilon = 1e-10
            prob_i = y_pred_score[:, i] + epsilon
            prob_j = y_pred_score[:, j] + epsilon

            # Compare classes i and j
            prob_ratios = prob_i / (prob_i + prob_j)
            i_wins = prob_ratios > thresholds[i][j]

            # Add votes
            votes[i_wins, i] += 1
            votes[~i_wins, j] += 1

    return np.argmax(votes, axis=1)
