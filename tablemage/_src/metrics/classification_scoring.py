import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


from ..display.print_utils import print_wrapped, color_text


class ClassificationBinaryScorer:
    """Class built for scoring of
    classification models that predict two classes.
    Only inputs are predicted and true values.
    Capable of scoring cross-validation outputs.
    """

    def __init__(
        self,
        y_pred: np.ndarray | list[np.ndarray],
        y_true: np.ndarray | list[np.ndarray],
        pos_label: float | int | str,
        y_pred_score: np.ndarray | list[np.ndarray] | None = None,
        name: str | None = None,
    ):
        """
        Initializes a ClassificationBinaryScorer object.

        Parameters
        ----------
        y_pred : np.ndarray ~ (sample_size) | list[np.ndarray ~ (sample_size)]

        y_true : np.ndarray ~ (sample_size) | list[np.ndarray ~ (sample_size)]

        pos_label : float | int | str
            The positive class label.

        y_pred_score : np.ndarray ~ (sample_size) | list[np.ndarray ~ (sample_size)]
            | None
            Default: None.

        name : str | None
            Default: None.
        """

        if name is None:
            self._name = "Model"
        else:
            self._name = name
        self._y_pred = np.asarray(y_pred)
        self._y_true = np.asarray(y_true)
        self._pos_label = pos_label

        if y_pred_score is not None:
            if isinstance(y_pred_score, np.ndarray):
                if len(y_pred_score.shape) == 2:
                    y_pred_score = y_pred_score[:, 1]
            elif isinstance(y_pred_score, list):
                if len(y_pred_score[0].shape) == 2:
                    y_pred_score = [elem[:, 1] for elem in y_pred_score]
        self._y_pred_score = np.asarray(y_pred_score)

        self._stats_df = None
        self._cv_stats_df = None
        self._set_stats_df()

    def _set_stats_df(self):
        """
        Creates statistics DataFrames given y_pred and y_true. If y_pred and
        y_true are lists, then the elements are treated as
        cross-validation folds, and the statistics are averaged
        across all folds.
        """
        y_pred = self._y_pred
        y_true = self._y_true
        y_pred_score = self._y_pred_score

        df = pd.DataFrame(columns=["Statistic", self._name])
        cvdf = pd.DataFrame(columns=["Fold", "Statistic", self._name])

        if isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray):
            df.loc[len(df)] = pd.Series(
                {"Statistic": "accuracy", self._name: accuracy_score(y_true, y_pred)}
            )
            df.loc[len(df)] = pd.Series(
                {"Statistic": "f1", self._name: f1_score(y_true, y_pred)}
            )
            df.loc[len(df)] = pd.Series(
                {
                    "Statistic": "precision",
                    self._name: precision_score(y_true, y_pred, zero_division=np.nan),
                }
            )
            df.loc[len(df)] = pd.Series(
                {"Statistic": "recall", self._name: recall_score(y_true, y_pred)}
            )
            if y_pred_score is not None:
                df.loc[len(df)] = pd.Series(
                    {
                        "Statistic": "roc_auc",
                        self._name: roc_auc_score(y_true, y_pred_score),
                    }
                )
            df.loc[len(df)] = pd.Series({"Statistic": "n_obs", self._name: len(y_pred)})

            self._stats_df = df.set_index("Statistic")

        elif isinstance(y_pred, list) and isinstance(y_true, list):
            for i, (y_pred_elem, y_true_elem) in enumerate(zip(y_pred, y_true)):
                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        "Statistic": "accuracy",
                        self._name: accuracy_score(y_true_elem, y_pred_elem),
                        "Fold": i,
                    }
                )

                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        "Statistic": "f1",
                        self._name: f1_score(y_true_elem, y_pred_elem),
                        "Fold": i,
                    }
                )

                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        "Statistic": "precision",
                        self._name: precision_score(
                            y_true_elem, y_pred_elem, zero_division=np.nan
                        ),
                        "Fold": i,
                    }
                )

                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        "Statistic": "recall",
                        self._name: recall_score(y_true_elem, y_pred_elem),
                        "Fold": i,
                    }
                )

                if y_pred_score is not None:
                    cvdf.loc[len(cvdf)] = pd.Series(
                        {
                            "Statistic": "roc_auc",
                            self._name: roc_auc_score(y_true_elem, y_pred_score[i]),
                            "Fold": i,
                        }
                    )

                cvdf.loc[len(cvdf)] = pd.Series(
                    {"Statistic": "n_obs", self._name: len(y_pred_elem), "Fold": i}
                )

            self._cv_stats_df = cvdf.set_index(["Fold", "Statistic"])
            self._stats_df = (
                cvdf.groupby(["Statistic"])[[self._name]]
                .mean()
                .reindex(["accuracy", "f1", "precision", "recall", "roc_auc", "n_obs"])
            )

    def stats_df(self) -> pd.DataFrame:
        """Outputs a DataFrame that contains the model's evaluation metrics.

        Returns
        -------
        pd.DataFrame
        """
        return self._stats_df

    def cv_stats_df(self) -> pd.DataFrame:
        """Outputs a DataFrame that contains the cross-validation
        evaluation metrics.

        Returns
        -------
        pd.DataFrame
        """
        return self._cv_stats_df

    def pos_label(self) -> str:
        """Outputs the positive class label.

        Returns
        -------
        str
        """
        return str(self._pos_label)


class ClassificationMulticlassScorer:
    """Class built for scoring of
    classification models that predict multiple classes.
    Only inputs are predicted and true values.
    Capable of scoring cross-validation outputs.
    """

    def __init__(
        self,
        y_pred: np.ndarray | list,
        y_true: np.ndarray | list,
        y_pred_score: np.ndarray | list | None = None,
        y_pred_class_order: np.ndarray | None = None,
        name: str | None = None,
    ):
        """
        Initializes a ClassificationMulticlassScorer object.

        Parameters
        ----------
        y_pred : np.ndarray ~ (sample_size) | list[np.ndarray ~ (sample_size)]

        y_true : np.ndarray ~ (sample_size) | list[np.ndarray ~ (sample_size)]

        y_pred_score : np.ndarray ~ (sample_size, n_classes) |
            list[np.ndarray ~ (sample_size, n_classes)] | None
            Default: None.

        y_pred_class_order : np.ndarray ~ (n_classes) | None
            Default: None

        name : str | None
            Default: None.
        """

        if name is None:
            self._name = "Model"
        else:
            self._name = name
        self._y_pred = np.asarray(y_pred)
        self._y_true = np.asarray(y_true)
        self._y_pred_score = np.asarray(y_pred_score)
        self._y_pred_class_order = np.asarray(y_pred_class_order)

        self._stats_df = None
        self._cv_stats_df = None
        self._set_stats_df()

        self._stats_by_class_df = None
        self._cv_stats_by_class_df = None
        self._set_stats_by_class_df()

    def _set_stats_df(self):
        """
        Creates statistics DataFrames given y_pred and y_true. If y_pred and
        y_true are lists, then the elements are treated as
        cross-validation folds, and the statistics are averaged
        across all folds.
        """

        y_pred = self._y_pred
        y_true = self._y_true
        y_pred_score = self._y_pred_score

        df = pd.DataFrame(columns=["Statistic", self._name])
        cvdf = pd.DataFrame(columns=["Fold", "Statistic", self._name])

        if isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray):
            df.loc[len(df)] = pd.Series(
                {"Statistic": "accuracy", self._name: accuracy_score(y_true, y_pred)}
            )
            df.loc[len(df)] = pd.Series(
                {
                    "Statistic": "f1",
                    self._name: f1_score(y_true, y_pred, average="macro"),
                }
            )
            df.loc[len(df)] = pd.Series(
                {
                    "Statistic": "precision",
                    self._name: precision_score(
                        y_true, y_pred, average="macro", zero_division=np.nan
                    ),
                }
            )
            df.loc[len(df)] = pd.Series(
                {
                    "Statistic": "recall",
                    self._name: recall_score(y_true, y_pred, average="macro"),
                }
            )
            if y_pred_score is not None:
                try:
                    df.loc[len(df)] = pd.Series(
                        {
                            "Statistic": "roc_auc(ovo)",
                            self._name: roc_auc_score(
                                y_true,
                                y_pred_score,
                                average="macro",
                                multi_class="ovo",
                                labels=self._y_pred_class_order,
                            ),
                        }
                    )
                except Exception as e:
                    print_wrapped(
                        "Error occured when computing the roc_auc "
                        + "score: "
                        + color_text(str(e), "blue"),
                        type="WARNING",
                    )
                    df.loc[len(df)] = pd.Series(
                        {"Statistic": "roc_auc", self._name: np.nan}
                    )
            df.loc[len(df)] = pd.Series({"Statistic": "n_obs", self._name: len(y_pred)})
            self._stats_df = df.set_index("Statistic")

        elif isinstance(y_pred, list) and isinstance(y_true, list):
            for i, (y_pred_elem, y_true_elem) in enumerate(zip(y_pred, y_true)):
                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        "Statistic": "accuracy",
                        self._name: accuracy_score(y_true_elem, y_pred_elem),
                        "Fold": i,
                    }
                )

                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        "Statistic": "f1",
                        self._name: f1_score(
                            y_true_elem,
                            y_pred_elem,
                            average="macro",
                            zero_division=np.nan,
                        ),
                        "Fold": i,
                    }
                )

                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        "Statistic": "precision",
                        self._name: precision_score(
                            y_true_elem,
                            y_pred_elem,
                            average="macro",
                            zero_division=np.nan,
                        ),
                        "Fold": i,
                    }
                )

                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        "Statistic": "recall",
                        self._name: recall_score(
                            y_true_elem,
                            y_pred_elem,
                            average="macro",
                            zero_division=np.nan,
                        ),
                        "Fold": i,
                    }
                )

                if y_pred_score is not None:
                    try:
                        cvdf.loc[len(cvdf)] = pd.Series(
                            {
                                "Statistic": "roc_auc(ovo)",
                                self._name: roc_auc_score(
                                    y_true_elem,
                                    y_pred_score[i],
                                    average="macro",
                                    multi_class="ovo",
                                    labels=self._y_pred_class_order,
                                ),
                                "Fold": i,
                            }
                        )
                    except Exception as e:
                        print_wrapped(
                            "Error occured when computing the roc_auc(ovo) "
                            + "score: "
                            + color_text(str(e), "blue"),
                            type="WARNING",
                        )
                        cvdf.loc[len(cvdf)] = pd.Series(
                            {"Statistic": "roc_auc(ovo)", self._name: np.nan, "Fold": i}
                        )

                cvdf.loc[len(cvdf)] = pd.Series(
                    {"Statistic": "n_obs", self._name: len(y_pred_elem), "Fold": i}
                )

            self._cv_stats_df = cvdf.set_index(["Fold", "Statistic"])
            self._stats_df = (
                cvdf.groupby(["Statistic"])[[self._name]]
                .mean()
                .reindex(
                    ["accuracy", "f1", "precision", "recall", "roc_auc(ovo)", "n_obs"]
                )
            )

        else:
            raise ValueError("Input types for y_pred and y_true are invalid.")

    def _set_stats_by_class_df(self):
        """Creates statistics DataFrames given y_pred and y_true. If y_pred and
        y_true are lists, then the elements are treated as
        cross-validation folds, and the statistics are averaged
        across all folds.

        Unlike _set_stats_df, this method creates DataFrames that contain
        statistics for each class, rather than overall (averaged) statistics
        across all classes.
        """

        df = pd.DataFrame(columns=["Class", "Statistic", self._name])
        cvdf = pd.DataFrame(columns=["Fold", "Class", "Statistic", self._name])

        if isinstance(self._y_pred, np.ndarray) and isinstance(
            self._y_true, np.ndarray
        ):
            try:
                roc_auc = roc_auc_score(
                    self._y_true, self._y_pred_score, multi_class="ovr", average=None
                )
            except Exception:
                roc_auc = np.nan
        elif isinstance(self._y_pred, list) and isinstance(self._y_true, list):
            try:
                roc_auc = [
                    roc_auc_score(
                        y_true_elem, y_pred_score_elem, multi_class="ovr", average=None
                    )
                    for y_true_elem, y_pred_score_elem in zip(
                        self._y_true, self._y_pred_score
                    )
                ]
            except Exception:
                roc_auc = np.nan

        for i, pos_label in enumerate(self._y_pred_class_order):
            if isinstance(self._y_pred, np.ndarray) and isinstance(
                self._y_true, np.ndarray
            ):
                df.loc[len(df)] = pd.Series(
                    {
                        "Class": pos_label,
                        "Statistic": "accuracy",
                        self._name: accuracy_score(self._y_true, self._y_pred),
                    }
                )
                df.loc[len(df)] = pd.Series(
                    {
                        "Class": pos_label,
                        "Statistic": "f1",
                        self._name: f1_score(
                            self._y_true,
                            self._y_pred,
                            labels=[pos_label],
                            average="macro",
                            zero_division=np.nan,
                        ),
                    }
                )
                df.loc[len(df)] = pd.Series(
                    {
                        "Class": pos_label,
                        "Statistic": "precision",
                        self._name: precision_score(
                            self._y_true,
                            self._y_pred,
                            labels=[pos_label],
                            average="macro",
                            zero_division=np.nan,
                        ),
                    }
                )
                df.loc[len(df)] = pd.Series(
                    {
                        "Class": pos_label,
                        "Statistic": "recall",
                        self._name: recall_score(
                            self._y_true,
                            self._y_pred,
                            labels=[pos_label],
                            average="macro",
                            zero_division=np.nan,
                        ),
                    }
                )
                if self._y_pred_score is not None:
                    if roc_auc is not np.nan:
                        roc_auc_elem = roc_auc[i]
                    else:
                        roc_auc_elem = np.nan
                    df.loc[len(df)] = pd.Series(
                        {
                            "Class": pos_label,
                            "Statistic": "roc_auc(ovr)",
                            self._name: roc_auc_elem,
                        }
                    )
                df.loc[len(df)] = pd.Series(
                    {
                        "Class": pos_label,
                        "Statistic": "n_obs",
                        self._name: len(self._y_pred),
                    }
                )
                self._stats_by_class_df = df.set_index(["Class", "Statistic"])

            elif isinstance(self._y_pred, list) and isinstance(self._y_true, list):
                for j, (y_pred_elem, y_true_elem) in enumerate(
                    zip(self._y_pred, self._y_true)
                ):
                    cvdf.loc[len(cvdf)] = pd.Series(
                        {
                            "Class": pos_label,
                            "Statistic": "accuracy",
                            self._name: accuracy_score(y_true_elem, y_pred_elem),
                            "Fold": j,
                        }
                    )
                    cvdf.loc[len(cvdf)] = pd.Series(
                        {
                            "Class": pos_label,
                            "Statistic": "f1",
                            self._name: f1_score(
                                y_true_elem,
                                y_pred_elem,
                                labels=[pos_label],
                                average="macro",
                                zero_division=np.nan,
                            ),
                            "Fold": j,
                        }
                    )
                    cvdf.loc[len(cvdf)] = pd.Series(
                        {
                            "Class": pos_label,
                            "Statistic": "precision",
                            self._name: precision_score(
                                y_true_elem,
                                y_pred_elem,
                                labels=[pos_label],
                                average="macro",
                                zero_division=np.nan,
                            ),
                            "Fold": j,
                        }
                    )
                    cvdf.loc[len(cvdf)] = pd.Series(
                        {
                            "Class": pos_label,
                            "Statistic": "recall",
                            self._name: recall_score(
                                y_true_elem,
                                y_pred_elem,
                                labels=[pos_label],
                                average="macro",
                                zero_division=np.nan,
                            ),
                            "Fold": j,
                        }
                    )
                    if self._y_pred_score is not None:
                        if roc_auc is not np.nan:
                            roc_auc_elem = roc_auc[j][i]
                        else:
                            roc_auc_elem = np.nan
                        cvdf.loc[len(cvdf)] = pd.Series(
                            {
                                "Class": pos_label,
                                "Statistic": "roc_auc(ovr)",
                                self._name: roc_auc_elem,
                                "Fold": j,
                            }
                        )
                    cvdf.loc[len(cvdf)] = pd.Series(
                        {
                            "Class": pos_label,
                            "Statistic": "n_obs",
                            self._name: len(y_pred_elem),
                            "Fold": j,
                        }
                    )
                self._cv_stats_by_class_df = cvdf.set_index(
                    ["Class", "Fold", "Statistic"]
                )
                self._stats_by_class_df = cvdf.groupby(["Class", "Statistic"])[
                    [self._name]
                ].mean()

    def stats_df(self) -> pd.DataFrame:
        """Outputs a DataFrame that contains the model's evaluation metrics.

        Returns
        -------
        pd.DataFrame
        """
        return self._stats_df

    def cv_stats_df(self) -> pd.DataFrame:
        """Outputs a DataFrame that contains the cross-validation
        evaluation metrics.

        Returns
        -------
        pd.DataFrame
        """
        return self._cv_stats_df

    def stats_by_class_df(self) -> pd.DataFrame:
        """Outputs a DataFrame that contains the model's evaluation metrics
        for each class.

        Returns
        -------
        pd.DataFrame
        """
        return self._stats_by_class_df

    def cv_stats_by_class_df(self) -> pd.DataFrame:
        """Outputs a DataFrame that contains the cross-validated
        evaluation metrics for each class.

        Returns
        -------
        pd.DataFrame
        """
        return self._cv_stats_by_class_df
