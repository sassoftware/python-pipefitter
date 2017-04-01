
.. Copyright SAS Institute

******************************
Installation and configuration
******************************

The pipefitter package installs just like any other Python package.
It is a pure Python package and works with both Python 2.7 and Python 3.4+
installations.  To install the latest version using ``pip``, you execute
the following::

    pip install pipefitter

You can also install previous releases from GitHub directly as follows::

    pip install https://github.com/sassoftware/pipefitter/releases/pipefitter-X.X.X.tar.gz

The releases for pipefitter can be found at
`<https://github.com/sassoftware/pipefitter/releases/>`_.

Remember that to be useful, the pipefitter package requires access to SAS 9.4 with
the SASPy package, or SAS Cloud Analytic Services with the SWAT package.

There is no configuration for the pipefitter package, but the related packages,
SASPy and SWAT have configuration steps to enable connections to SAS 9.4 and
SAS Cloud Analytic Services, respectively.
