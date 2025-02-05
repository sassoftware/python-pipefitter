This project is no longer under active development and was archived on 2025-02-05.

**************
SAS Pipefitter
**************

The SAS Pipefitter project provides a Python API for developing
machine learning pipelines. The pipelines are built from stages
that perform variable transformation, parameter estimation, and
hyperparameter tuning. 

A key feature of pipefitter is that the pipelines can run
with a SAS 9.4 platform as well as with SAS Cloud Analytic
Services (CAS) in the SAS Viya platform.

This package builds on the SASPy and SAS SWAT packages. Those
packages enable Python to work with SAS 9.4 and SAS Viya,
respectively.

Installation
============

SAS pipefitter can be installed as follows with `pip`::

    pip install pipefitter

Alternatively, you can install a specific release as follows::

    pip install https://github.com/sassoftware/python-pipefitter/releases/download/vX.X.X/pipefitter-X.X.X.tar.gz

Dependencies
============

The package is a pure Python package and works with Python 2.7 or 3.4+.

However, the packages that enable Python to work with the different
SAS platforms introduce dependencies of their own.


+--------------+-------------------+--------------------+
| Package      | SAS Platfom       | Python Support     | 
+==============+===================+====================+
| SASPy 2.x    | SAS 9.4 or higher | Python 3.x         | 
+--------------+-------------------+--------------------+
| SWAT v1.1.0+ | SAS Viya          | Python 2.7 or 3.4+ |
+--------------+-------------------+--------------------+


Resources
=========

`Pipefitter documentation <http://sassoftware.github.io/python-pipefitter/>`_

`SASPy <http://github.com/sassoftware/saspy/>`_

`SAS SWAT for Python <http://github.com/sassoftware/python-swat/>`_

`Python <http://www.python.org/>`_
