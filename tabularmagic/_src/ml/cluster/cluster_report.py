from typing import Literal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ...data.datahandler import DataHandler
from ...display.plot_options import plot_options
from .base_cluster import BaseClust


class ClusterReport:
    """Class for reporting clustering results.
    Fits models based on provided DataHandler.
    """

    def __init__(
        self,
        models: list[BaseClust],
        datahandler: DataHandler,
        features: list[str],
        dataset: Literal["train", "all"],
    ):
        """Initializes ClusterReport.

        Parameters
        ----------
        models : list[BaseClust]
            List of models to fit.

        datahandler : DataHandler
            DataHandler object.

        features : list[str]
            List of feature names.

        dataset : Literal["train", "all"]
            Dataset to fit models on.
            If "train", only fits models on training data.
            Then, predictions can be made on test data.
            If "all", fits models on all data.
        """
        self._models = models

        self._id_to_model = {}
        for model in self._models:
            if model._id in self._id_to_model:
                raise ValueError(
                    f"Model IDs must be unique. Duplicate found: {model._id}"
                )
            self._id_to_model[model._id] = model

        for model in self._models:
            if not isinstance(model, BaseClust):
                raise TypeError("All models must be of type BaseCluster.")

        self._X_vars = features

        if dataset == "train":
            self._emitter = datahandler.train_test_emitter(
                y_var=None,
                X_vars=self._X_vars,
            )

        elif dataset == "all":
            self._emitter = datahandler.full_dataset_emitter(
                y_var=None,
                X_vars=self._X_vars,
            )

        else:
            raise ValueError("dataset must be 'train' or 'all'.")

        for model in self._models:
            model.specify_data(self._emitter)
            model.fit()

    def model(self, model_id: str) -> BaseClust:
        """Returns model by ID.

        Parameters
        ----------
        model_id : str
            Model ID.

        Returns
        -------
        BaseClust
            Model with specified ID.
        """
        if model_id not in self._id_to_model:
            raise ValueError(f"Model {model_id} not found.")
        return self._id_to_model[model_id]

    def metrics(self) -> pd.DataFrame:
        """Returns DataFrame with model metrics.

        Returns
        -------
        pd.DataFrame
            DataFrame with model metrics.
        """
        raise NotImplementedError("Metrics not yet implemented.")
        # metrics = []
        # for model in self._models:
        #     metrics.append(model.metrics())
        # return pd.concat(metrics, axis=1)

    def plot_clusters(
        self,
        model_id: str,
        dim_reduction_method: Literal["PCA", "TSNE"] = "PCA",
        dataset: Literal["train", "test"] = "test",
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plots clusters for specified model.

        Parameters
        ----------
        model_id : str
            Model ID to obtain labels from.

        dim_reduction_method: Literal["PCA", "TSNE"]
            Dimensionality reduction method.
            Default is "PCA".

        dataset : Literal["train", "test"]
            Dataset to plot. If the ClusterReport was initialized with
            dataset="all", then both "train" and "test" yield
            the same results (test is same as train in this case). Default is "test".

        figsize : tuple[float, float]
            Figure size. Default is (5, 5).

        ax : plt.Axes | None
            Axes to plot on. If None, a new figure is created. Default is None.

        Returns
        -------
        plt.Figure
            Figure with cluster plot.
        """
        X_df: pd.DataFrame = None
        if dataset == "train":
            X_df = self._emitter.emit_train_X()
        elif dataset == "test":
            X_df = self._emitter.emit_test_X()
        else:
            raise ValueError("dataset must be 'train' or 'test'.")
        labels_series = self.model(model_id).labels(dataset)

        X_reduced = None

        if dim_reduction_method == "PCA":
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X_df)
            X_reduced = pd.DataFrame(
                X_reduced, columns=["PC1", "PC2"], index=X_df.index
            )

        elif dim_reduction_method == "TSNE":
            tsne = TSNE(n_components=2)
            X_reduced = tsne.fit_transform(X_df)
            X_reduced = pd.DataFrame(
                X_reduced, columns=["t-SNE1", "t-SNE2"], index=X_df.index
            )

        else:
            raise ValueError("dim_reduction_method must be 'PCA' or 'TSNE'.")

        plotting_df = X_reduced.join(labels_series)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        sns.scatterplot(
            x=plotting_df.columns[0],
            y=plotting_df.columns[1],
            hue=labels_series.name,
            data=plotting_df,
            palette=plot_options._color_palette,
            ax=ax,
        )

        ax.set_title(f"{model_id} Clusters")
        ax.set_xlabel(plotting_df.columns[0])
        ax.set_ylabel(plotting_df.columns[1])

        ax.title.set_fontsize(plot_options._title_font_size)
        ax.xaxis.label.set_fontsize(plot_options._axis_title_font_size)
        ax.yaxis.label.set_fontsize(plot_options._axis_title_font_size)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=plot_options._axis_major_ticklabel_font_size,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=plot_options._axis_minor_ticklabel_font_size,
        )

        legend = ax.legend_
        if legend is not None:
            legend.set_title(
                legend.get_title().get_text(),
                prop={"size": plot_options._axis_title_font_size},
            )
            for text in legend.get_texts():
                text.set_fontsize(plot_options._axis_title_font_size)

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig
