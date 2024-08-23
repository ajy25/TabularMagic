Introduction
============

.. currentmodule:: tabularmagic

Domain expertise is extremely valuable in data science and quantitative research.
However, experts in fields such as medicine and healthcare often lack the 
necessary statistical and computational experience to analyze data on their own.

TabularMagic simplifies data science, making it
accessible to researchers with little to no quantitative background. 
TabularMagic empowers domain experts to perform their own data
analysis and machine learning modeling without needing to learn the intricacies 
of data science (Python, pandas, scikit-learn, ML theory). 
TabularMagic's Python API is designed to be intuitive and easy to use, with 
low-code functions that abstract away the complexity of data science. 

Below, we list the steps of a typical data science workflow and describe
how TabularMagic can help at each step. 

.. rubric:: Data Loading and Cleaning

Data must be extracted and cleaned prior to analysis. 
Unfortunately, TabularMagic does not handle data loading and cleaning. 
TabularMagic works with wide format tabular data in the form of a pandas DataFrame.
See :class:`tabularmagic.Analyzer` for details.


.. rubric:: Data Exploration

New datasets need to be explored prior to further processing and modeling.
Data scientists working with Python typically use pandas and matplotlib/seaborn
to explore data. 

TabularMagic enhances the data exploration process by providing a 
comprehensive report object that includes the following:

1. Categorical and numeric variable summary statistics
2. Low-code statistical testing
3. Low-code visualizations

See :meth:`tabularmagic.Analyzer.eda` for details.


.. rubric:: Data Preprocessing

Before modeling, data must be preprocessed (e.g., missing values imputed, 
categorical variables encoded, etc.). 

TabularMagic streamlines the data preprocessing process, reducing
several minutes of documentation reading and 10+ lines of code down to zero minutes of 
documentation reading and one line of code. 

See :meth:`tabularmagic.Analyzer.impute`, 
:meth:`tabularmagic.Analyzer.scale`, 
:meth:`tabularmagic.Analyzer.drop_highly_missing_vars`, 
:meth:`tabularmagic.Analyzer.dropna`,
and :meth:`tabularmagic.Analyzer.onehot` for details. 

.. note::
    Categorical variables are automatically one-hot encoded in the modeling process
    if you do not one-hot encode them beforehand. Also, observations with missing values 
    are automatically dropped in the modeling process. 


.. rubric:: Statistical Modeling

Statistical modeling is sometimes preferred over machine learning modeling
because it provides interpretable results.

TabularMagic improves the statistical modeling process by providing low-code 
functions for linear regression and generalized linear models. 

See 
:meth:`tabularmagic.Analyzer.lm` and :meth:`tabularmagic.Analyzer.glm` for details.


.. rubric:: Machine Learning Modeling

Machine learning modeling is often preferred over statistical modeling
because it can handle more complex relationships in the data.

TabularMagic simplifies the machine learning modeling process by providing
low-code functions for classification and regression models. 

See
:meth:`tabularmagic.Analyzer.classify` and :meth:`tabularmagic.Analyzer.regress` for 
details. The different models available are listed in 
:doc:`../python_api/classes/tm_ml`.





