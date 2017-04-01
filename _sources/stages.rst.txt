
.. Copyright SAS Institute

.. currentmodule:: pipefitter

####################################################
Working with transformers, estimators, and pipelines
####################################################


**********************
Using basic estimators
**********************

The following simple example begins with using SAS SWAT to connect to
SAS Cloud Analytic Services on SAS Viya.

.. ipython:: python
   :suppress:

   import os
   host = os.environ['CASHOST']
   port = os.environ['CASPORT']
   userid = None
   password = None
   cfgname = os.environ.get('SASPY_CONFIG', 'tdi')

.. ipython:: python

   import swat
   conn = swat.CAS(host, port, userid, password)

All processing in pipefitter begins with a data set.  In the case of SAS SWAT,
the data set is a :class:`CASTable <swat.cas.table.CASTable>` object.  There are many ways to create
a :class:`CASTable` object, but one of the easiest is the 
:meth:`CAS.upload_file` method.

.. ipython:: python

   tbl = conn.upload_file('http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv')
   tbl.head()

To start with, we train a decision tree model.  This is done using the 
:class:`DecisionTree` object in the :mod:`pipefitter.estimator` module.
We set the ``target``, ``inputs``, and ``nominals`` arguments for the previously
loaded table.

.. ipython:: python

   from pipefitter.estimator import DecisionTree

   dtree = DecisionTree(target='Survived',
                        inputs=['Sex', 'Age', 'Fare'],
                        nominals=['Sex', 'Survived'])
   dtree

We can now train the model using the :meth:`DecisionTree.fit` method.  This
method returns a :class:`DecisionTreeModel` object.

.. ipython:: python

   model = dtree.fit(tbl)
   model

The :meth:`DecisionTreeModel.score` method can then be called on the resulting
model object.  In this case, we can use the same data table as before.

.. ipython:: python

   score = model.score(tbl)
   score

The result of the :meth:`DecisionTreeModel.score` method is a :class:`pandas.Series`.
The rows in this result vary depending on the backend associated with the 
data set.  We can start a SAS 9 session using SASPy to see what it generates.

Note that when switching backends, we can use the same :class:`DecisionTree` object,
but we have to re-train the model.  Each backend stores model information in its 
own way.

.. ipython:: python

   import saspy
   sas9 = saspy.SASsession(cfgname=cfgname)

   ds = sas9.read_csv('http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv',
                      'titanic')

   dtree.fit(ds).score(ds)

As you can see, the SASPy backend returns more information, but some fields such as
``NObsUsed`` and ``MisClassificationRate`` are shared between them.

In addition to :class:`DecisionTree`, there are other estimators such as
:class:`DecisionForest` and :class:`GBTree` that can be used in similar ways.
You can also use these estimators in pipelines.

*********
Pipelines
*********

Pipelines can be used to construct a series of data processing steps into a
repeatable workflow.  These steps commonly include imputation, feature selection,
and feature creation.  We can create imputers to fill the missing values in our data
then use them in a pipeline.

We can use the ``info`` method on the data set to first see which columns contain
missing values.

.. ipython:: python

   tbl.info()

There are both numeric and character columns that contain missing values.
We can create a mode imputer to fill missing character values, and a mean
imputer for the numeric missing values.

.. ipython:: python
   
   from pipefitter.transformer import Imputer

   modeimp = Imputer(value=Imputer.MODE)
   meanimp = Imputer(value=Imputer.MEAN)

Applying both of these imputers to our data using the ``transform`` method
results in a data set with no missing values.

.. ipython:: python

   outtbl = modeimp.transform(tbl)
   outtbl = meanimp.transform(outtbl)
   outtbl.info()

We can now add these imputers to a pipeline before an estimator class to create
a self-contained workflow.

.. ipython:: python

   from pipefitter.pipeline import Pipeline

   pipe = Pipeline([modeimp, meanimp, dtree])
   pipe

Just as with an estimator class, you call the ``fit`` and ``score`` methods
on the :class:`Pipeline` and :class:`PipelineModel` objects, respectively.

.. ipython:: python

   model = pipe.fit(tbl)
   model

.. ipython:: python

   score = model.score(tbl)
   score

The same pipeline can be executed on the SAS 9 data set as well.

.. ipython:: python

   pipe.fit(ds).score(ds)


*********************
HyperParameter Tuning
*********************

In addition to creating pipelines of transformers and estimators, you can test
various permutations of parameters using the :class:`HyperParameterTuning` class.
This class takes a grid of parameters to test and applies them to an estimator
or a pipeline and returns the compiled results.  The parameter grid can be either
a dictionary of key/value pairs where the values are lists, or a list of dictionaries
containing the complete set of parameters to test.

.. ipython:: python
   :okwarning:

   from pipefitter.model_selection import HyperParameterTuning as HPT

   hpt = HPT(estimator=dtree,
             param_grid=dict(max_depth=[6, 10],
                             leaf_size=[3, 5]),
             score_type='MisClassificationRate',
             cv=3) 
   hpt.gridsearch(tbl, n_jobs=4)

As previously mentioned, hyperparameter tuning can also be done on pipelines.

.. ipython:: python
   :okwarning:

   hpt = HPT(estimator=pipe,
             param_grid=dict(max_depth=[6, 10],
                             leaf_size=[3, 5]),
             score_type='MisClassificationRate',
             cv=3)
   hpt.gridsearch(tbl, n_jobs=4)


.. ipython:: python
   :suppress:

   conn.endsession()
   conn.close()
