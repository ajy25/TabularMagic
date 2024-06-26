import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
from typing import Iterable
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc


import seaborn as sns

sns.set_theme()


def plot_obs_vs_pred(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    figsize: Iterable = (5, 5),
    ax: axes.Axes = None,
) -> plt.Figure:
    """Returns a figure that is a scatter plot of the observed and predicted y
    values. Predicted values on x axis, observed values on y axis.

    Parameters
    ----------
    y_pred : np.ndarray.
        The predicted y values.
    y_true : np.ndarray.
        The observed y values.
    figsize : Iterable.
        Default: (5, 5). The size of the figure. Only used if ax is None.
    ax : plt.Axes.
        Default: None. The axes to plot on. If None, a new figure is created.

    Returns
    -------
    plt.Figure.
        Figure of the scatter plot.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    min_val = np.min(np.hstack((y_pred, y_true)))
    max_val = np.max(np.hstack((y_pred, y_true)))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="gray",
        linewidth=1,
    )

    ax.scatter(y_pred, y_true, s=2, color="black")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")

    ax.set_title(
        "Observed vs Predicted | " + f"ρ = {round(pearsonr(y_pred, y_true)[0], 3)}"
    )
    ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3))
    ax.yaxis.get_offset_text().set_fontsize(ax.yaxis.get_ticklabels()[0].get_fontsize())
    ax.xaxis.get_offset_text().set_fontsize(ax.xaxis.get_ticklabels()[0].get_fontsize())

    if fig is not None:
        fig.tight_layout()
        plt.close()
    return fig


def plot_roc_curve(
    y_score: np.ndarray,
    y_true: np.ndarray,
    figsize: Iterable = (5, 5),
    ax: axes.Axes = None,
):
    """Returns a figure that is the ROC curve for the model.

    Parameters
    ----------
    y_score : np.ndarray.
        Predicted probabilities or decision scores.
    y_true : np.ndarray.
        True binary labels.
    figsize : Iterable.
        Default: (5, 5). The size of the figure. Only used if ax is None.
    ax : plt.Axes.
        Default: None. The axes to plot on. If None, a new figure is created.

    Returns
    -------
    plt.Figure.
        Figure of the ROC curve.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.plot(fpr, tpr, color="black")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve | AUC = {roc_auc:.3f}")

    if fig is not None:
        fig.tight_layout()
        plt.close()
    return fig


def decrease_font_sizes_axs(
    axs,
    title_font_size_decrease: int,
    axes_label_font_size_decrease: int,
    ticks_font_size_decrease_from_label: int = 0,
):
    """
    Decreases the font sizes of titles, axes labels, and ticks in a set of axes.

    Parameters
    ----------
    axs.
    title_font_size_decrease : int.
    axes_label_font_size_decrease : int.
    ticks_font_size_decrease_from_label : int.
    """
    for ax in axs.flat:
        ax.title.set_fontsize(ax.title.get_fontsize() - title_font_size_decrease)
        ax.xaxis.label.set_fontsize(
            ax.xaxis.label.get_fontsize() - axes_label_font_size_decrease
        )
        ax.yaxis.label.set_fontsize(
            ax.yaxis.label.get_fontsize() - axes_label_font_size_decrease
        )
        tick_params = ax.tick_params(
            axis="both",
            which="major",
            labelsize=ax.xaxis.label.get_fontsize()
            - ticks_font_size_decrease_from_label,
        )
        if tick_params is not None:
            tick_params.set_fontsize(
                tick_params.get("size") - ticks_font_size_decrease_from_label
            )

        new_offset_size_diff = (
            axes_label_font_size_decrease + ticks_font_size_decrease_from_label
        )
        ax.xaxis.offsetText.set_fontsize(
            ax.xaxis.offsetText.get_fontsize() - new_offset_size_diff
        )
        ax.yaxis.offsetText.set_fontsize(
            ax.yaxis.offsetText.get_fontsize() - new_offset_size_diff
        )
