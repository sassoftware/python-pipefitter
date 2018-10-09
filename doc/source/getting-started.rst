
.. Copyright SAS Institute

.. currentmodule:: pipefitter

###############
Getting started
###############


******************
About this example
******************

This section provides a simple example that demonstrates how to use the
basic functionality of the software to create repeatable pipelines. The
simplest example follows this pattern:

#. Create an instance of a :class:`Pipeline` class.
#. Add transformer stages to handle variable imputation.
#. Add an estimator stage to generate parameter estimates.
#. Run the pipeline with the :meth:`fit` method to create a pipeline model.
#. Run the :meth:`score` method on the model with new data to score and 
   assess the model.

To demonstrate these common steps for developing a machine learning pipeline,
the Titanic training data set from a Kaggle competition is used.

Because pipefitter can run in SAS 9 or SAS Viya, the last two steps are data
dependent and this document shows how to run the pipeline first with SAS 9
and then with SAS Viya.


.. ipython:: python
   :suppress:

   import os
   host = os.environ['CASHOST']
   port = os.environ['CASPORT']
   userid = os.environ.get('CASUSER', None)
   password = os.environ.get('CASPASSWORD', None)
   cfgname = os.environ.get('SASPY_CONFIG', 'tdi')


******************
Build the pipeline
******************

First, download the training data to a Pandas DataFrame.

.. ipython:: python
   
   import pandas as pd
   train = pd.read_csv('http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv')
   train.head()

There are both numeric and character columns that contain missing values.
The pipeline can start with two transformer stages to fill missing numeric
values with the mean and missing character values with the most common value.

.. ipython:: python
   
   from pipefitter.transformer import Imputer

   meanimp = Imputer(value=Imputer.MEAN)
   modeimp = Imputer(value=Imputer.MODE)

The following statements add these stages to a pipeline. Printing the object
shows that the pipeline includes the two stages.  

.. ipython:: python

   from pipefitter.pipeline import Pipeline
   pipe = Pipeline([meanimp, modeimp])
   pipe

The last stage of the pipeline is an estimator. To model the survival of 
passengers, we can train a decision tree model. This is done using the 
:class:`DecisionTree` object in the :mod:`pipefitter.estimator` module.
We set the target, inputs, and nominals arguments to match the 
variables in the data set.

.. ipython:: python

   from pipefitter.estimator import DecisionTree

   dtree = DecisionTree(target='Survived',
                        inputs=['Sex', 'Age', 'Fare'],
                        nominals=['Sex', 'Survived'])
   dtree

In addition to :class:`DecisionTree`, there are other estimators such as
:class:`DecisionForest`, :class:`GBTree`, and :class:`LogisticRegression` 
that can be used in similar ways. You can use these estimators in pipelines.

To complete the work on the pipeline, add the estimator stage.

.. ipython:: python

   pipe.stages.append(dtree)
   
   for stage in pipe:
       print(stage, "\n")
   
Now that the pipeline is complete, we just need to add training data.


***************************
Run the pipeline in SAS 9.4
***************************

This section follows on the pipeline set up work from the preceding section.
The SASPy package enables you to run analytics in SAS 9.4 and higher from
a Python API.  

First, start a SAS session and copy the training data to a SAS data set.

.. ipython:: python

   import saspy

   sasconn = saspy.SASsession(cfgname=cfgname)

   train_ds = sasconn.df2sd(df=train, table="train_ds")
   train_ds.head()

.. note:: For information about starting a SAS session with the SASPy package,
          see http://sassoftware.github.io/saspy/getting-started.html#start-a-sas-session.

Next, generate the model using the :meth:`Pipeline.fit` method. The training
data is supplied as an argument. This method returns a :class:`PipelineModel` 
object.

.. ipython:: python

   pipeline_model = pipe.fit(train_ds)

   for stage in pipeline_model:
       print(stage, "\n")


.. note:: In the pipeline model, the decision tree becomes a decision
          tree model.

View the model assessment by running the :meth:`PipelineModel.score` method
with the training data.

.. ipython:: python

   pipeline_model.score(train_ds)


************************************************************
Run the pipeline in SAS Viya and SAS Cloud Analytic Services
************************************************************


This section follows on the pipeline set up work from the first section. If
you ran the SAS 9.4 section with SASPy, you can continue with the code in
this section:

* The pipelines are designed to be portable.
* The location of the data determines where the pipeline runs--in SAS 9.4
  or in-memory tables in CAS.
* The model implementations are different between platforms, so the model
  does need to be retrained.

Use SAS SWAT to connect to SAS Cloud Analytic Services on SAS Viya.

.. ipython:: python

   import swat
   casconn = swat.CAS(host, port, userid, password)

.. note:: For information about starting a CAS session with the SWAT package,
          see https://sassoftware.github.io/python-swat/getting-started.html.

All processing in pipefitter begins with a data set.  In the case of SAS SWAT,
the data set is a :class:`CASTable <swat.cas.table.CASTable>` object. You can
create a CASTable from the training data that was initially downloaded.

.. ipython:: python

   train_ct = casconn.upload_frame(train, casout=dict(name="train_ct", replace=True))

   train_ct.info()

The results of the info function demonstrate again that the training
data has missing values. Because the pipeline has the two transformer stages, the
missing values are replaced when the model is trained.

Use the same :class:`Pipeline` instance to generate another :class:`PipelineModel`.

.. ipython:: python

   pipeline_model = pipe.fit(train_ct)

   for stage in pipeline_model:
     print(stage, "\n")

Use the pipeline model to score the training data and show the model assessment.

.. ipython:: python

   pipeline_model.score(train_ct)
   
As you can see, the results include many measures that are in common with the 
results from SAS 9.4 with SASPy. For example, the ``NObsUsed`` and 
``MisClassificationRate`` values are common.


****************************
Bonus: HyperParameter tuning
****************************

In addition to creating pipelines of transformers and estimators, you can test
various permutations of parameters using the :class:`HyperParameterTuning` class.
This class takes a grid of parameters to test and applies them to an estimator
or a pipeline and returns the compiled results.  The parameter grid can be either
a dictionary of key/value pairs where the values are lists, or a list of dictionaries
containing the complete set of parameters to test.

The hyperparameter tuning can be performed on pipelines.

.. ipython:: python
   :okwarning:

   from pipefitter.model_selection import HyperParameterTuning as HPT

   hpt = HPT(estimator=pipe,
             param_grid=dict(max_depth=[6, 10],
                             leaf_size=[3, 5]),
             score_type='MisClassificationRate',
             cv=3)
   hpt.gridsearch(train_ct)


.. ipython:: python
   :suppress:

   casconn.endsession()
   casconn.close()
