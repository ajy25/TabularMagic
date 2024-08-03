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

:py:mod:`tm._reports.PoissonRegressionReport`
---------------------------------------------

.. autoclass:: PoissonRegressionReport
    :members:
        __init__, model, metrics, step, statsmodels_summary


:py:mod:`tm._reports.LogisticRegressionReport`
----------------------------------------------

.. autoclass:: LinearRegressionReport
    :members:
        __init__, model, metrics, step, statsmodels_summary


