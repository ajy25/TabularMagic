import matplotlib.pyplot as plt
plt.ioff()
import numpy as np

def plot_predicted_vs_true_scatter(y_pred: np.ndarray, y_true: np.ndarray) \
        -> plt.Figure:
    """Returns a simple predicted vs true scatter plot figure. 
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=2, color='black')
    min_val = np.min(np.hstack((y_pred, y_true)))
    max_val = np.max(np.hstack((y_pred, y_true)))
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    return fig
