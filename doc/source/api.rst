
.. Copyright SAS Institute

.. currentmodule:: pipefitter
.. _api:

*************
API Reference
*************

Estimators
----------

Estimators are one of the stages that are added to a pipeline.
An estimator is typically the last stage in a pipeline.

When you add an estimator, such as a :class:`DecisionTree` or
:class:`LinearRegression` to a pipeline, the ``fit`` method
performs parameter estimation and generates a model. The pipeline
returns a :class:`PipelineModel` that includes the model.

.. currentmodule:: pipefitter.estimator

Decision Tree
~~~~~~~~~~~~~

.. autosummary::
      :toctree: generated/

   DecisionTree
   DecisionTree.fit

Decision Forest
~~~~~~~~~~~~~~~

.. autosummary::
      :toctree: generated/

   DecisionForest
   DecisionForest.fit

Gradient Boosting Tree
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
      :toctree: generated/

   GBTree
   GBTree.fit

Logistic Regression
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   LogisticRegression
   LogisticRegression.fit

Linear Regression
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   LinearRegression
   LinearRegression.fit

Neural Network
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   NeuralNetwork
   NeuralNetwork.fit


Models
------

The model classes are created as a result of adding
an estimator class to a :class:`Pipeline` and running
the ``fit`` method with training data.  The pipeline
returns a :class:`PipelineModel` that includes the model
form of the estimator.  For example, if the pipeline 
included a :class:`NeuralNetwork` estimator, then the
returned pipeline model includes a :class:`NeuralNetworkModel`
instance.

Decision Tree Model
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pipefitter.estimator.tree

.. autosummary::
         :toctree: generated/

   DecisionTreeModel.score
   DecisionTreeModel.transform

Decision Forest Model
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pipefitter.estimator.forest

.. autosummary::
         :toctree: generated/

   DecisionForestModel.score
   DecisionForestModel.transform

Gradient Boosting Tree Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pipefitter.estimator.gradient_boosting

.. autosummary::
         :toctree: generated/

   GBTreeModel.score
   GBTreeModel.transform

Logistic Regression Model
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pipefitter.estimator.regression

.. autosummary::
   :toctree: generated/

   LogisticRegressionModel.score
   LogisticRegressionModel.transform

Linear Regression Model
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pipefitter.estimator.regression

.. autosummary::
   :toctree: generated/

   LinearRegressionModel.score
   LinearRegressionModel.transform

Neural Network Model
~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pipefitter.estimator.neural_net

.. autosummary::
   :toctree: generated/

   NeuralNetworkModel.score
   NeuralNetworkModel.transform


Pipelines
---------

.. currentmodule:: pipefitter.pipeline

Pipeline
~~~~~~~~

Pipelines are a series of transformers followed by an estimator
used to construct a self-contained workflow.  When the ``fit`` or
``score`` method of the pipeline is executed, each stage of the 
pipeline is executed in order.  The output of each stage is used
as the input for the next stage.  The result is the output from 
the last stage.

.. autosummary::
            :toctree: generated/

   Pipeline
   Pipeline.fit
   Pipeline.transform

Pipeline Model
~~~~~~~~~~~~~~

.. autosummary::
            :toctree: generated/

   PipelineModel.score
   PipelineModel.transform


Transformers
------------

Transformers are used to modify your data sets.  Currently, this
includes various forms of imputing missing values in your data sets.

Imputers
~~~~~~~~

.. currentmodule:: pipefitter.transformer.imputer

.. autosummary::
            :toctree: generated/

   Imputer
   Imputer.transform


HyperParameter Tuning
---------------------

Hyperparameter tuning allows you to test various combinations of
model parameters in one workflow.  This can be done either on
a single Estimator class instance or a Pipeline.

.. currentmodule:: pipefitter.model_selection

.. autosummary::
            :toctree: generated/

   HyperParameterTuning
   HyperParameterTuning.gridsearch
