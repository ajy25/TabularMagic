Reports :py:mod:`(tm._reports)`
===============================

Report objects are outputted by the :meth:`.lm`, :meth:`.glm`, :meth:`.regress`, 
and :meth:`.classify` methods of the :class:`Analyzer` class. They contain
information about the model's performance, feature importance, and other
relevant statistics. They also have methods for plotting relevant diagnostic figures.


.. currentmodule:: tabularmagic._reports


:py:mod:`tm._reports.MLClassificationReport`
--------------------------------------------

.. autoclass:: MLClassificationReport
    :members:
        __init__, model, metrics, cv_metrics, fs_report, plot_confusion_matrix, plot_roc_curve, metrics_by_class, cv_metrics_by_class 


:py:mod:`tm._reports.MLRegressionReport`
----------------------------------------

.. autoclass:: MLRegressionReport
    :members:
        __init__, model, metrics, cv_metrics, fs_report, plot_obs_vs_pred

:py:mod:`tm._reports.LinearRegressionReport`
---------------------------------------------

.. autoclass:: LinearRegressionReport
    :members:
        __init__, train_report, test_report, model, metrics, step, test_lr,
        test_partialf, statsmodels_summary, plot_obs_vs_pred, 
        plot_residuals_vs_fitted, plot_residuals_vs_var, plot_residuals_hist,
        plot_scale_location, plot_residuals_vs_leverage, plot_qq, 
        plot_diagnostics, set_outlier_threshold, get_outlier_indices,
        _compute_outliers


:py:mod:`tm._reports.PoissonRegressionReport`
----------------------------------------------

.. autoclass:: PoissonRegressionReport
    :members:
        __init__, train_report, test_report, model, metrics, step, 
        statsmodels_summary, plot_obs_vs_pred, plot_residuals_vs_fitted, 
        plot_residuals_vs_var, plot_residuals_hist, plot_scale_location,
        plot_residuals_vs_leverage, plot_qq, plot_diagnostics, 
        set_outlier_threshold, get_outlier_indices, _compute_outliers


:py:mod:`tm._reports.NegativeBinomialRegressionReport`
-------------------------------------------------------

.. autoclass:: NegativeBinomialRegressionReport
    :members:
        __init__, train_report, test_report, model, metrics, step, 
        statsmodels_summary, plot_obs_vs_pred, plot_residuals_vs_fitted, 
        plot_residuals_vs_var, plot_residuals_hist, plot_scale_location,
        plot_residuals_vs_leverage, plot_qq, plot_diagnostics, 
        set_outlier_threshold, get_outlier_indices, _compute_outliers


:py:mod:`tm._reports.BinomialRegressionReport`
----------------------------------------------

.. autoclass:: BinomialRegressionReport
    :members:
        __init__, train_report, test_report, model, metrics, step, 
        statsmodels_summary, plot_obs_vs_pred, plot_residuals_vs_fitted, 
        plot_residuals_vs_var, plot_residuals_hist, plot_scale_location,
        plot_residuals_vs_leverage, plot_qq, plot_diagnostics, 
        set_outlier_threshold, get_outlier_indices, _compute_outliers


:py:mod:`tm._reports.CountRegressionReport`
----------------------------------------------

.. autoclass:: CountRegressionReport
    :members:
        __init__, train_report, test_report, model, metrics, step, 
        statsmodels_summary, plot_obs_vs_pred, plot_residuals_vs_fitted, 
        plot_residuals_vs_var, plot_residuals_hist, plot_scale_location,
        plot_residuals_vs_leverage, plot_qq, plot_diagnostics, 
        set_outlier_threshold, get_outlier_indices, _compute_outliers


:py:mod:`tm._reports.EDAReport`
-------------------------------

.. autoclass:: EDAReport
    :members:
        __init__, plot_distribution, plot_distribution_stratified, plot_numeric_pairs, plot_pca, test_equal_means, anova, ttest, numeric_vars, categorical_vars, categorical_stats, numeric_stats


:py:mod:`tm._reports.VotingSelectionReport`
-------------------------------------------

.. autoclass:: VotingSelectionReport
    :members:
        __init__, top_features, all_features, votes

