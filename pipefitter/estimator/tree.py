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
Decision Tree

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import functools
from ..base import BaseEstimator, BaseModel
from ..utils.params import (param_def, check_int, check_string, check_boolean,
                            check_float, check_variable, check_variable_list,
                            extract_params)


class DecisionTree(BaseEstimator):
    '''
    Decision Tree

    Parameters
    ----------
    alpha : float, optional
        Specifies the value to use for minimal cost-complexity pruning
        for regression trees
    cf_level : float, optional
        Specifies the aggressiveness of tree pruning according to the C4.5
        algorithm.
    criterion : string, optional
        Split criterion. Valid values are 'variance', 'gain', 'gain_ratio',
        and 'gini'.
    leaf_size : int, optional
        Minimum leaf size
    max_braches : int, optional
        Maximum number of branches
    max_depth : int, optional
        Maximum depth of trees
    n_bins : int, optional
        Number of bins to use for numeric variables in the calculation
        of the decision tree
    prune : boolean, optional
        Specify True to use a C4.5 pruning method for classification
        trees or minimal cost-complexity pruning for regression trees
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
    >>> dtree = DecisionTree(target='Origin',
    ...                      inputs=['MPG_City', 'MPG_Highway', 'Length',
    ...                              'Weight', 'Type', 'Cylinders'],
    ...                      nominals = ['Type', 'Cylinders', 'Origin'])

    Returns
    -------
    :class:`DecisionTree`

    '''

    param_defs = dict(
        alpha=param_def(0, functools.partial(check_float, minimum=0)),
        cf_level=param_def(0.25, functools.partial(check_float, minimum=0, maximum=1)),
        criterion=param_def(None, functools.partial(check_string, allow_none=True,
                                      valid_values=['variance', 'gain',
                                                    'gain_ratio', 'gini'])),
        leaf_size=param_def(5, functools.partial(check_int, minimum=1)),
        max_branches=param_def(2, functools.partial(check_int, minimum=1)),
        max_depth=param_def(6, functools.partial(check_int, minimum=1)),
        n_bins=param_def(20, functools.partial(check_int, minimum=1)),
        prune=param_def(False, check_boolean),
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
        stat=False,
    )

    def __init__(self, alpha=0, cf_level=0.25, criterion=None, leaf_size=5,
                       max_branches=2, max_depth=6, n_bins=20, prune=False,
                       var_importance=False, target=None, nominals=None, inputs=None):
        BaseEstimator.__init__(self, **extract_params(locals()))
        
    def fit(self, table, *args, **kwargs):
        ''' 
        Fit function for decision tree

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
        >>> dtree = DecisionTree(target='Origin',
        ...                      inputs=['MPG_City', 'MPG_Highway', 'Length',
        ...                              'Weight', 'Type', 'Cylinders'],
        ...                      nominals = ['Type', 'Cylinders', 'Origin'])
        >>> model = dtree.fit(training_data)

        Returns
        -------
        :class:`DecisionTreeModel`

        '''
        params = self.get_combined_params(*args, **kwargs)
        return self._get_super(table).fit(table, **params)


class DecisionTreeModel(BaseModel):
    ''' DecisionTree trained model '''

    param_defs = DecisionTree.param_defs
    static_params = DecisionTree.static_params
