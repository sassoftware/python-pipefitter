
.. Copyright SAS Institue

##########################
Overview of SAS pipefitter
##########################


***********************
What is SAS pipefitter?
***********************

The SAS pipefitter package provides a Python API for developing
pipelines for data transformation and model fitting as stages 
of a repeatable workflow. 

.. image:: images/pipeline.png


The package enables you to work with data in SAS to implement 
stages such as the following:

* Impute missing values.
* Select and engineer features.
* Fit parameter estimates with decision trees, neural networks,
  and other machine learning techniques.
* Use hyperparameter tuning to speed model selection.
* Score data and assess models.

Another important feature of pipefitter is that it builds on the
abilities of two other Python packages provided by SAS:

SWAT
  A package that enables data transfer and analysis with 
  SAS Cloud Analytic Services--the centerpiece of the SAS Viya
  platform--for in-memory analytics.

SASPy
  A package that enables data transfer and analysis with
  SAS 9.4--business-proven software for analytics, data
  manipulation, and visualization.

The pipeline stages, such as estimating parameters with logistic
regression are designed to run identically in SAS 9 though SASPy
or in CAS through SWAT.  The pipeline automatically adjusts to run the 
analysis where your data is by detecting the data set type.
:class:`SASdata` objects will run through SASPy, whereas :class:`CASTable`
objects run through SWAT.

See :doc:`getting-started` for examples.


************
Dependencies
************

SAS pipefitter is a pure Python package that supports Python 2.7 or 3.4+.
However, you must install one or both of the following packages that
are used to connect to SAS 9.4 or SAS Viya.

+--------------------------------------------------------------+-------------------+--------------------+
| Package                                                      | SAS Platfom       | Python Support     |
+==============================================================+===================+====================+
| `SASPy 2.x <https://github.com/sassoftware/saspy>`_          | SAS 9.4 or higher | Python 3.x         |
+--------------------------------------------------------------+-------------------+--------------------+
| `SWAT v1.1.0+ <https://github.com/sassoftware/python-swat>`_ | SAS Viya          | Python 2.7 or 3.4+ |
+--------------------------------------------------------------+-------------------+--------------------+

Read the package documentation for additional dependencies, supported
platforms, and so on.

.. tip:: The most common combination that provides the 
         greatest flexibility is to use Linux with 64-bit 
         Python 3.4+.


**************
Project status
**************

Release 1.0.0 of SAS pipefitter represents an early stage of 
development. This release exploits only a small portion of what
is possible with SAS 9.4 and SAS Viya.

For more information, see the following:

* https://www.sas.com/en_us/software/sas9.html
* https://www.sas.com/en_us/software/viya.html
