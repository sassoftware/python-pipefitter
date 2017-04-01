#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

'''
Regression

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import functools
from ..base import BaseEstimator, BaseModel
from ..utils.params import (param_def, check_int, check_string, check_boolean,
                            check_float, check_variable, check_variable_list)


class LogisticRegression(BaseEstimator):
    '''
    Logistic Regression

    Parameters
    ----------
    intercept : bool, optional
        Include the intercept term in the model?
    max_effects : int, optional
        Specifies the maximum number of effects in any model to consider 
        during the selection process
    selection : string, optional
        Specifies the selection method. Valid values are 'none', 'backward',
        'forward', and 'stepwise'.
    sig_level : float, optional
        Specifies the significance level
    criterion : string, optional
        Specifies selection criterion. Valid values are 'sl', 'aic', 'aicc',
        and 'sbc'.
    target : string, optional
        The target variable
    nominals : string or list of strings, optional
        The nominal variables
    inputs : string or list of strings, optional
        The input variables

    Examples
    --------
    >>> log = LogisticRegression(target='Origin',
    ...                          inputs=['MPG_City', 'MPG_Highway', 'Length',
    ...                                  'Weight', 'Type', 'Cylinders'],
    ...                          nominals = ['Type', 'Cylinders', 'Origin'])

    Returns
    -------
    :class:`LogisticRegression`

    '''

    param_defs = dict(
        intercept=param_def(True, check_boolean),
        max_effects=param_def(0, functools.partial(check_int, minimum=0)),
        selection=param_def('none', functools.partial(check_string,
                                    valid_values=['none', 'backward',
                                                  'forward', 'stepwise'])),
        sig_level=param_def(0.05, functools.partial(check_float, minimum=0.0, maximum=1.0)),
        criterion=param_def(None, functools.partial(check_string, allow_none=True,
                                  valid_values=['sl', 'aic', 'aicc', 'sbc'])),
        target=param_def(None, check_variable),
        nominals=param_def(None, check_variable_list),
        inputs=param_def(None, check_variable_list),
    )

    def __init__(self, intercept=True, max_effects=0, selection='none',
                       sig_level=0.05, criterion=None,  
                       target=None, nominals=None, inputs=None):
        BaseEstimator.__init__(self, intercept=intercept, max_effects=max_effects,
                               selection=selection, sig_level=sig_level, criterion=criterion,
                               target=target, nominals=nominals, inputs=inputs)
        if self.params['criterion'] == 'sl' and \
           self.params['selection'] in ['backward', 'lasso']:
               raise ValueError("criterion='sl' is not valid with "
                                "selection='backward' | 'lasso'")
        
    def fit(self, table, *args, **kwargs):
        ''' 
        Fit function for logistic regression

        Parameters
        ----------
        *args : dicts or two-element tuples or consecutive key/value pairs, optional
            The following types are allowed:
                * Dictionaries contain key/value pairs of parameters.  
                * Two-element tuples must contain the name of the parameter in the
                  first element and the value in the second element.  
                * Consecutive key/value pairs are also allowed.
        **kwargs : keyword arguments, optional
            These keyword arguments are the same as on the constructor.

        Examples
        --------
        >>> log = LogisticRegression(target='Origin',
        ...                          inputs=['MPG_City', 'MPG_Highway', 'Length',
        ...                                  'Weight', 'Type', 'Cylinders'],
        ...                          nominals = ['Type', 'Cylinders', 'Origin'])
        >>> model = log.fit(training_data)

        Returns
        -------
        :class:`LogisticRegressionModel`

        '''
        params = self.get_combined_params(*args, **kwargs)
        return self._get_super(table).fit(table, **params)


class LogisticRegressionModel(BaseModel):
    ''' LogisticRegresson trained model '''

    param_defs = LogisticRegression.param_defs


class LinearRegression(BaseEstimator):
    '''
    Linear Regression

    Parameters
    ----------
    intercept : bool, optional
        Include the intercept term in the model?
    max_effects : int, optional
        Specifies the maximum number of effects in any model to consider
        during the selection process
    selection : string, optional
        Specifies the selection method. Valid values are 'none', 'backward',
        'forward', 'lasso', and 'stepwise'.
    sig_level : float, optional
        Specifies the significance level
    criterion : string, optional
        Specifies selection criterion. Valid values are 'sl', 'aic', 'aicc',
        and 'sbc'.
    target : string, optional
        The target variable
    nominals : string or list of strings, optional
        The nominal variables
    inputs : string or list of strings, optional
        The input variables

    Examples
    --------
    >>> lin = LinearRegression(target='MSRP',
    ...                        inputs=['MPG_City', 'MPG_Highway', 'Length',
    ...                                'Weight', 'Type', 'Cylinders'],
    ...                        nominals = ['Type', 'Cylinders', 'Origin'])

    Returns
    -------
    :class:`LinearRegression`

    '''

    param_defs = dict(
        intercept=param_def(True, check_boolean),
        max_effects=param_def(0, functools.partial(check_int, minimum=0)),
        selection=param_def('none', functools.partial(check_string,
                                    valid_values=['none', 'backward', 'forward',
                                                  'lasso', 'stepwise'])),
        sig_level=param_def(0.05, functools.partial(check_float, minimum=0.0, maximum=1.0)),
        criterion=param_def(None, functools.partial(check_string, allow_none=True,
                                  valid_values=['sl', 'aic', 'aicc', 'sbc'])),
        target=param_def(None, check_variable),
        nominals=param_def(None, check_variable_list),
        inputs=param_def(None, check_variable_list),
    )

    def __init__(self, intercept=True, max_effects=0, selection='none',
                       sig_level=0.05, criterion=None,
                       target=None, nominals=None, inputs=None):
        BaseEstimator.__init__(self, intercept=intercept, max_effects=max_effects,
                               selection=selection, sig_level=sig_level, criterion=criterion,
                               target=target, nominals=nominals, inputs=inputs)
        if self.params['criterion'] == 'sl' and \
           self.params['selection'] in ['backward', 'lasso']:
               raise ValueError("criterion='sl' is not valid with "
                                "selection='backward' | 'lasso'")
        
    def fit(self, table, *args, **kwargs):
        '''
        Fit function for linear regression

        Parameters
        ----------
        *args : dicts or two-element tuples or consecutive key/value pairs, optional
            The following types are allowed:
                * Dictionaries contain key/value pairs of parameters.
                * Two-element tuples must contain the name of the parameter in the
                  first element and the value in the second element.
                * Consecutive key/value pairs are also allowed.
        **kwargs : keyword arguments, optional
            These keyword arguments are the same as on the constructor.

        Examples
        --------
        >>> lin = LinearRegression(target='MSRP',
        ...                        inputs=['MPG_City', 'MPG_Highway', 'Length',
        ...                                'Weight', 'Type', 'Cylinders'],
        ...                        nominals = ['Type', 'Cylinders', 'Origin'])
        >>> model = lin.fit(training_data)

        Returns
        -------
        :class:`LinearRegressionModel`

        '''
        params = self.get_combined_params(*args, **kwargs)
        return self._get_super(table).fit(table, **params)


class LinearRegressionModel(BaseModel):
    ''' LinearRegresson trained model '''

    param_defs = LinearRegression.param_defs
