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
Gradient Boosting Tree

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import functools
from ..utils.params import (check_int, check_boolean, check_float, check_string,
                            check_variable, check_variable_list, param_def,
                            extract_params)
from ..base import BaseEstimator, BaseModel


class GBTree(BaseEstimator):
    ''' 
    Gradient Boosting Tree

    Parameters
    ----------
    distribution : string, optional
        Split criterion. Valid values are 'gaussian', 'binary', and 'multinomial'.
    early_stop_stagnation : int, optional
        Early stop stagnation parameter
    lasso : float, optional
        Specifies the L1 norm regularization on prediction
    leaf_size : int, optional
        Minimum leaf size
    learning_rate : float, optional
        Specifies the learning rate of each tree
    m : int, optional
        Specifies the number of variables in each split
    max_braches : int, optional
        Maximum number of branches
    max_depth : int, optional
        Maximum depth of trees
    n_bins : int, optional
        Number of bins to use for numeric variables in the calculation
        of the decision tree
    n_trees : int, optional
        Specifies the number of trees to create
    ridge : float, optional
        Specifies the L2 norm regularization on prediction
    seed : float, optional
        Specifies the seed for the random number generator
    subsample_rate : float, optional
        Specifies the fraction of the data to use for building each tree
    var_importannce : bool, optional
        Specifies whether the variable importance information is
        generated
    target : string, optional
        The target variable
    nominals : string or list of strings, optional
        The nominal variables
    inputs : string or list of strings, optional
        The input variables

    Examples
    --------
    >>> gbt = GBTree(target='Origin',
    ...              inputs=['MPG_City', 'MPG_Highway', 'Length',
    ...                      'Weight', 'Type', 'Cylinders'],
    ...              nominals = ['Type', 'Cylinders', 'Origin'])

    Returns
    -------
    :class:`GBTree`

    '''

    param_defs = dict(
        distribution=param_def(None, functools.partial(check_string, allow_none=True,
                                         valid_values=['gaussian', 'binary',
                                                       'multinomial'])),
        early_stop_stagnation=param_def(0, functools.partial(check_int, minimum=0)),
        lasso=param_def(0, functools.partial(check_float, minimum=0)),
        leaf_size=param_def(5, functools.partial(check_int, minimum=1)),
        learning_rate=param_def(0.1, functools.partial(check_float, minimum=0,
                                                                    maximum=1)),
        m=param_def(None, functools.partial(check_int, minimum=1, allow_none=True)),
        max_branches=param_def(2, functools.partial(check_int, minimum=1)),
        max_depth=param_def(6, functools.partial(check_int, minimum=1)),
        n_bins=param_def(20, functools.partial(check_int, minimum=1)),
        n_trees=param_def(50, functools.partial(check_int, minimum=1)),
        ridge=param_def(0, functools.partial(check_float, minimum=0)),
        seed=param_def(0, functools.partial(check_float, minimum=0)),
        subsample_rate=param_def(0.5, functools.partial(check_float, minimum=0,
                                                                     maximum=1)),
        var_importance=param_def(False, check_boolean),
        target=param_def(None, check_variable),
        nominals=param_def(None, check_variable_list),
        inputs=param_def(None, check_variable_list),
    )

    static_params = dict(
        bin_order=True,
        greedy=True,
        merge_bin=True,
        include_missing=True,
        missing='use_in_search',
    )

    def __init__(self, distribution=None, early_stop_stagnation=0, lasso=0,
                       leaf_size=5, learning_rate=0.1, m=None, max_branches=2,
                       max_depth=6, n_bins=20, n_trees=50, ridge=0, 
                       seed=0, subsample_rate=0.5, var_importance=False,
                       target=None, nominals=None, inputs=None):
        BaseEstimator.__init__(self, **extract_params(locals()))

    def fit(self, table, *args, **kwargs):
        '''
        Fit function for gradient boosting tree

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
        >>> gbt = GBTree(target='Origin',
        ...              inputs=['MPG_City', 'MPG_Highway', 'Length',
        ...                      'Weight', 'Type', 'Cylinders'],
        ...              nominals = ['Type', 'Cylinders', 'Origin'])
        >>> model = gbt.fit(training_data)

        Returns
        -------
        :class:`GBTreeModel`

        '''
        params = self.get_combined_params(*args, **kwargs)
        return self._get_super(table).fit(table, **params)


class GBTreeModel(BaseModel):
    ''' Gradient boosting tree trained model '''

    param_defs = GBTree.param_defs
    static_params = GBTree.static_params
