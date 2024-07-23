tm.DataHandler (:py:mod:`tabularmagic.Analyzer`)
================================================


.. currentmodule:: tabularmagic

The DataHandler class handles data loading, saving, and preprocessing 
in the background for the Analyzer class.


.. autoclass:: DataHandler
    :members: load_data_checkpoint, save_data_checkpoint, remove_data_checkpoint, df_all, df_train, df_test, vars, numvars, catvars, head, scaler, train_test_emitter, kfold_emitters, dropna, onehot, drop_highly_missing_vars, impute, scale, add_scaler, select_vars, drop_vars, force_numerical, force_binary, force_categorical