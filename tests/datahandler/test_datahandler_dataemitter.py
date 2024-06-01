import unittest
import pathlib
import sys
parent_dir = str(pathlib.Path(__file__).resolve().parent.parent.parent)
sys.path.append(parent_dir)

import pandas as pd
import numpy as np

from tabularmagic._src.data.datahandler import (DataEmitter, DataHandler, 
                                                PreprocessStepTracer)



from sklearn.model_selection import train_test_split
from sklearn import datasets



class TestDataEmitter(unittest.TestCase):

    def setUp(self):
        self.df_simple = pd.DataFrame(
            {
                'binary_var': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                'categorical_var': ['A', 'B', 'C', 'A', 'B', 
                                    'C', 'A', 'B', 'C', 'A'],
                'continuous_var': [5.2, 3.7, 8.1, 2.5, 6.9, 
                                   4.3, 7.8, 1.1, 3.4, 5.7]
            }
        )
        self.df_simple_train, self.df_simple_test = train_test_split(
            self.df_simple, test_size=0.2, random_state=42
        )
        iris = datasets.load_iris()
        data = pd.DataFrame(iris.data, columns=iris.feature_names)
        target = pd.Series(iris.target)
        self.df_iris = pd.concat([data, target], axis=1)
        self.df_iris.columns.values[-1] = 'target'

        self.df_iris_train, self.df_iris_test = train_test_split(
            self.df_iris, test_size=0.2, random_state=42
        )


    def test_basic_init(self):
        """Test basic initialization DataEmitter functionality."""

        dh = DataHandler(self.df_simple_train, self.df_simple_test, 
                         verbose=False)
        de = DataEmitter(self.df_simple_train, self.df_simple_test, 
                         'binary_var', ['categorical_var', 'continuous_var'], 
                         PreprocessStepTracer())
        dh_emitter = dh.train_test_emitter(
                'binary_var', ['categorical_var', 'continuous_var']
            )
        self.assertEqual(
            dh_emitter._working_df_test.shape,
            de._working_df_test.shape
        )
        self.assertEqual(
            dh_emitter._working_df_train.shape,
            de._working_df_train.shape
        )
        dh.drop_vars(['binary_var'])
        self.assertNotEqual(
            dh._working_df_train.shape,
            de._working_df_train.shape
        )

    
    def test_force_categorical(self):
        """Test force categorical encoding of continuous or binary
        variables."""

        dh = DataHandler(self.df_simple_train, self.df_simple_test, 
                         verbose=False)
        de = DataEmitter(self.df_simple_train, self.df_simple_test, 
                         'binary_var', ['categorical_var', 'continuous_var'], 
                         PreprocessStepTracer())
        dh.force_categorical(['binary_var'])
        dh_emitter = dh.train_test_emitter('binary_var', 
            ['categorical_var', 'continuous_var'])
        self.assertEqual(
            dh_emitter._working_df_train['binary_var'].dtype,
            'object'
        )
        self.assertEqual(
            dh_emitter._working_df_test.shape,
            de._working_df_test.shape
        )
        self.assertEqual(
            dh_emitter._working_df_train.shape,
            de._working_df_train.shape
        )


    def test_force_continuous(self):
        """Test force continuous encoding of categorical variables."""

        dh = DataHandler(self.df_simple_train, self.df_simple_test, 
                         verbose=False)
        de = DataEmitter(self.df_simple_train, self.df_simple_test, 
                         'binary_var', ['categorical_var', 'continuous_var'], 
                         PreprocessStepTracer())
        dh.force_continuous(['binary_var'])
        dh_emitter = dh.train_test_emitter('binary_var', 
            ['categorical_var', 'continuous_var'])
        self.assertEqual(
            dh_emitter._working_df_train['binary_var'].dtype,
            'float64'
        )
        self.assertEqual(
            dh_emitter._working_df_test.shape,
            de._working_df_test.shape
        )
        self.assertEqual(
            dh_emitter._working_df_train.shape,
            de._working_df_train.shape
        )
    

    def test_force_binary(self):
        """Test force binary encoding of continuous or categorical 
        variables."""

        dh = DataHandler(self.df_simple_train, self.df_simple_test, 
                         verbose=False)
        de = DataEmitter(self.df_simple_train, self.df_simple_test, 
                         'binary_var', ['categorical_var', 'continuous_var'], 
                         PreprocessStepTracer())
        dh.force_binary(['continuous_var'])
        dh_emitter = dh.train_test_emitter('binary_var', 
            ['categorical_var', 'continuous_var'])
        self.assertEqual(
            dh_emitter._working_df_train['continuous_var'].dtype,
            de._working_df_train['continuous_var'].dtype
        )
        self.assertEqual(
            dh_emitter._working_df_test.shape,
            de._working_df_test.shape
        )
        self.assertEqual(
            dh_emitter._working_df_train.shape,
            de._working_df_train.shape
        )

        dh = DataHandler(self.df_simple_train, self.df_simple_test, 
                         verbose=False)
        de = DataEmitter(self.df_simple_train, self.df_simple_test, 
                         'binary_var', ['categorical_var', 'continuous_var'], 
                         PreprocessStepTracer())
        de._force_binary(['categorical_var'], pos_labels=['A'], 
                        ignore_multiclass=True)
        dh.force_binary(['categorical_var'], pos_labels=['A'], 
                        ignore_multiclass=True)
        dh_emitter = dh.train_test_emitter('binary_var', 
            ['A_TRUE(categorical_var)', 'continuous_var'])
        self.assertEqual(
            dh_emitter._working_df_train['A_TRUE(categorical_var)'].dtype,
            de._working_df_train['A_TRUE(categorical_var)'].dtype
        )
        self.assertEqual(
            dh_emitter._working_df_test.shape,
            de._working_df_test.shape
        )
        self.assertEqual(
            dh_emitter._working_df_train.shape,
            de._working_df_train.shape
        )


        dh.force_binary(['continuous_var'], pos_labels=[5.2],
                        ignore_multiclass=True)
        dh_emitter = dh.train_test_emitter('binary_var', 
            ['A_TRUE(categorical_var)', '5.2_TRUE(continuous_var)'])
        self.assertEqual(
            dh._working_df_train['5.2_TRUE(continuous_var)'].dtype,
            'int64'
        )
        self.assertEqual(
            dh_emitter._working_df_train.shape,
            dh._working_df_train.shape
        )




    def test_onehot(self):
        """Test onehot encoding of categorical variables.
        """

        dh = DataHandler(self.df_simple_train, self.df_simple_test, 
                         verbose=False)
        dh.onehot(['categorical_var'], dropfirst=False)
        self.assertTrue(
            ('A_TRUE(categorical_var)' in dh._working_df_train.columns) and \
            ('B_TRUE(categorical_var)' in dh._working_df_train.columns) and \
            ('C_TRUE(categorical_var)' in dh._working_df_train.columns)
        )
        self.assertTrue(
            ('A_TRUE(categorical_var)' in dh._working_df_test.columns) and \
            ('B_TRUE(categorical_var)' in dh._working_df_test.columns) and \
            ('C_TRUE(categorical_var)' in dh._working_df_test.columns)
        )


        dh_emitter = dh.train_test_emitter('binary_var', 
            ['A_TRUE(categorical_var)', 'B_TRUE(categorical_var)', 
             'C_TRUE(categorical_var)', 'continuous_var'])
        self.assertTrue(
            ('A_TRUE(categorical_var)' in \
                dh_emitter._working_df_test.columns) and \
            ('B_TRUE(categorical_var)' in \
                dh_emitter._working_df_test.columns) and \
            ('C_TRUE(categorical_var)' in \
                dh_emitter._working_df_test.columns)
        )
        self.assertTrue(
            ('A_TRUE(categorical_var)' in \
                dh_emitter._working_df_train.columns) and \
            ('B_TRUE(categorical_var)' in \
                dh_emitter._working_df_train.columns) and \
            ('C_TRUE(categorical_var)' in \
                dh_emitter._working_df_train.columns)
        )
        self.assertEqual(
            dh_emitter._working_df_test.shape,
            dh._working_df_test.shape
        )
        self.assertEqual(
            dh_emitter._working_df_train.shape,
            dh._working_df_train.shape
        )



        dh = DataHandler(self.df_simple_train, self.df_simple_test, 
                         verbose=False)
        dh.onehot(['categorical_var', 'binary_var'], dropfirst=True)
        self.assertTrue(
            ('B_TRUE(categorical_var)' in dh._working_df_train.columns) and \
            ('C_TRUE(categorical_var)' in dh._working_df_train.columns)
        )
        self.assertTrue(
            ('B_TRUE(categorical_var)' in dh._working_df_test.columns) and \
            ('C_TRUE(categorical_var)' in dh._working_df_test.columns)
        )
        self.assertFalse(
            'A_TRUE(categorical_var)' in dh._working_df_train.columns
        )

        dh_emitter = dh.train_test_emitter('continuous_var', 
            ['B_TRUE(categorical_var)', 
             'C_TRUE(categorical_var)', '1_TRUE(binary_var)'])
        self.assertTrue(
            ('B_TRUE(categorical_var)' in \
                dh_emitter._working_df_test.columns) and \
            ('C_TRUE(categorical_var)' in \
                dh_emitter._working_df_test.columns)
        )
        self.assertTrue(
            ('B_TRUE(categorical_var)' in \
                dh_emitter._working_df_train.columns) and \
            ('C_TRUE(categorical_var)' in \
                dh_emitter._working_df_train.columns)
        )
        self.assertEqual(
            dh_emitter._working_df_test.shape,
            dh._working_df_test.shape
        )
        self.assertEqual(
            dh_emitter._working_df_train.shape,
            dh._working_df_train.shape
        )



    def test_multiple(self):
        """Test multiple preprocessing steps."""
        dh = DataHandler(self.df_iris_train, self.df_iris_test, 
                        verbose=False)
        dh.force_categorical(['target'])
        dh.scale(['sepallength(cm)'], strategy='log')
        dh.drop_vars(['sepalwidth(cm)'])
        dh_emitter = dh.train_test_emitter(
            'target', ['sepallength(cm)', 'petallength(cm)', 'petalwidth(cm)']
        )
        for col1, col2 in zip(dh_emitter._working_df_train.columns, 
                              dh._working_df_train.columns):
            self.assertEqual(
                col1, col2
            )
        self.assertEqual(
            dh_emitter._working_df_train.shape,
            dh._working_df_train.shape
        )
        for idx1, idx2 in zip(dh_emitter._working_df_train.index,
                                dh._working_df_train.index):
            self.assertEqual(
                idx1, idx2
            )


    def test_kfold_basic_init(self):
        """Test kfold cross validation basic functionality."""
        dh = DataHandler(self.df_iris_train, self.df_iris_test, 
                        verbose=False)
        dh.force_categorical(['target'])
        dh.scale(['sepallength(cm)'], strategy='log')
        dh.drop_vars(['sepalwidth(cm)'])
        emitters = dh.kfold_emitters(
            'target', ['sepallength(cm)', 'petallength(cm)', 'petalwidth(cm)']
        )
        idxs = []
        for emitter in emitters:
            for col1, col2 in zip(emitter._working_df_train.columns, 
                                  dh._working_df_train.columns):
                self.assertEqual(
                    col1, col2
                )
            idxs.append(emitter._working_df_train.index.to_list())
        idxs = np.concatenate(idxs)
        for idx in idxs:
            self.assertTrue(
                idx in dh._working_df_train.index
            )

        dh = DataHandler(self.df_iris_train, self.df_iris_test, 
            verbose=False)
        dh.force_categorical(['target'])
        dh.onehot(['target'])
        dh.scale(['sepallength(cm)'], strategy='minmax')
        dh.drop_vars(['sepalwidth(cm)'])
        emitters = dh.kfold_emitters(
            'sepallength(cm)', ['1_TRUE(target)', '2_TRUE(target)', 
                                'petallength(cm)', 'petalwidth(cm)'],
            n_folds=5,
            shuffle=True,
            random_state=42
        )
        idxs = []
        for emitter in emitters:
            for col1, col2 in zip(emitter._working_df_train.columns, 
                                  dh._working_df_train.columns):
                self.assertEqual(
                    col1, col2
                )
        idxs.append(emitter._working_df_train.index.to_list())
        idxs = np.concatenate(idxs)
        for idx in idxs:
            self.assertTrue(
                idx in dh._working_df_train.index
            )
        self.assertEqual(emitters[0].y_scaler().min, 4.3)
        self.assertNotEqual(emitters[2].y_scaler().min, 4.3)



if __name__ == '__main__':
    unittest.main()


