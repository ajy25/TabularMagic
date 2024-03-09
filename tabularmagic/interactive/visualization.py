import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable
from scipy.stats import pearsonr


def plot_pred_vs_true(y_pred: np.ndarray, y_true: np.ndarray, 
                      figsize: Iterable = (5, 5)):
    """Returns a figure that is a scatter plot of the true and predicted y 
    values. 

    Parameters 
    ----------
    - y_pred : np.ndarray
    - y_true : np.ndarray
    - figsize : Iterable. 

    Returns
    -------
    - plt.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(y_true, y_pred, s=2, color='black')
    min_val = np.min(np.hstack((y_pred, y_true)))
    max_val = np.max(np.hstack((y_pred, y_true)))
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    ax.set_title(f'Predicted vs True | ' + \
                    f'ρ = {round(pearsonr(y_pred, y_true)[0], 3)}')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(-2, 2))
    fig.tight_layout()
    plt.close()
    return fig








