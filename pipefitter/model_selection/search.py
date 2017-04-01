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
Model Selection Search

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import collections
import copy
import functools
import itertools
import numpy as np
import numbers
import pandas as pd
import time
import warnings

#from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
from ..base import BaseGridSearchCV, BaseEstimator, get_super_module
from ..pipeline import Pipeline
from ..utils.params import (check_string, check_int, check_number, param_def,
                            check_number_or_iter)

def param_iter(params):
    '''
    Iterate over all combinations of parameters

    params : dict or list of dicts
        The sets of parameters

    Returns
    -------
    dict

    '''
    if isinstance(params, collections.Mapping):
        keys, values = zip(*params.items())
        values = list(values)
        for i, val in enumerate(values):
            if not isinstance(val, collections.Sequence):
                values[i] = [val]
        for val in itertools.product(*values):
            yield dict(zip(keys, copy.deepcopy(val)))
    else:
        for value in params:
            yield copy.deepcopy(value)

def _fit_and_score(args):
    '''
    Create a new session then fit and score the given tables 
    
    Parameters
    ----------
    estimator : Estimator
    train_table : data set
    score_table : data set
    params : dict
    
    Returns
    -------
    DataFrame
    
    '''
    rmgr, estimator, train_table, score_table, parallel, params = args

    start_time = time.time()

    if parallel:
        train_table, score_table = rmgr.emancipate(train_table, score_table)

    model = estimator.fit(train_table, **params)
    rmgr.track_model(model)
    res = model.score(score_table)

    if res is None:
        return

    if isinstance(res, pd.Series):
        res = pd.DataFrame(res).T
        res['ClockTime'] = time.time() - start_time
        res['FitParameters'] = [params]
    else:
        res['ClockTime'] = time.time() - start_time
        res['FitParameters'] = [params] * len(res)

    return res


class HyperParameterTuning(BaseGridSearchCV):
    '''
    Perform search over all combinations of specified parameters
    
    Parameters
    ----------
    estimator : estimator
        The estimator class/object to use for fitting
    param_grid : dict or list of dicts
        The combinations of parameters to use.
            * dict - each key in the dictionary corresponds to a
              parameter name.  The values in the dictionary are
              lists of the parameter values to use.
            * list of dicts - each dictionary is a set of 
              parameters to use.
    score_type : string
        The score value to use in each iteration.  The default is
        'MisClassificationRate' for targets that are class
        variables, or 'AverageSquaredError' for targets that are
        interval variables.
    cv : int or float or generator, optional
        Indicates the cross validation folding scheme.
            * int - indicates the number of folds to apply to
              the data set.
            * float - indicates that one fold should be applied.
              The value is the percentage of observations to use
              for the training data set.
            * generator - specifies a generator that will return
              training and scoring data sets.

    Examples
    --------

    Using a dict of parameter lists:

    >>> hpt = HyperParameterTuning(estimator=estimator,
    ...                            param_grid = dict(
    ...                                max_depth=[6, 10],
    ...                                leaf_size=[3, 5]
    ...                            ))

    Using a list of parameter dictionaries:

    >>> hpt = HyperParameterTuning(estimator=estimator,
    ...                            param_grid = [
    ...                                dict(max_depth=6, leaf_size=3),
    ...                                dict(max_depth=6, leaf_size=5),
    ...                                dict(max_depth=10, leaf_size=3),
    ...                                dict(max_depth=10, leaf_size=5),
    ...                            ])

    Returns
    -------
    :class:`HyperParameterTuning`

    '''

    param_defs = dict(
        estimator=param_def(None, None),
        param_grid=param_def(None, None),
        score_type=param_def(None, functools.partial(check_string, allow_none=True)),
        n_jobs=param_def(1, functools.partial(check_int, minimum=1)),
        cv=param_def(3, functools.partial(check_number_or_iter, minimum_int=2,
                                          minimum_float=0.0, maximum_float=1.0)),
    )

    def gridsearch(self, table, n_jobs=None):
        '''
        Fit model over various permutations of parameters

        Parameters
        ----------
        table : data set
            The data set to use for training and scoring
        n_jobs : int
            The number of jobs to run in parallel (when supported by
            the backend)

        Notes
        -----
        For small data sets when n_jobs > 1, the overhead of creating 
        threads and multiple sessions on the backend may be greater than
        the time it takes to run each step sequentially.

        Examples
        --------

        Using a dict of parameter lists:

        >>> hpt = HyperParameterTuning(estimator=estimator,
        ...                            param_grid = dict(
        ...                                max_depth=[6, 10],
        ...                                leaf_size=[3, 5]
        ...                            ))
        >>> scores = hpt.gridsearch(data)

        Using a list of parameter dictionaries:

        >>> hpt = HyperParameterTuning(estimator=estimator,
        ...                            param_grid = [
        ...                                dict(max_depth=6, leaf_size=3),
        ...                                dict(max_depth=6, leaf_size=5),
        ...                                dict(max_depth=10, leaf_size=3),
        ...                                dict(max_depth=10, leaf_size=5),
        ...                            ])
        >>> scores = hpt.gridsearch(data)

        Returns
        -------
        :class:`DataFrame`

        '''
        try:
            ResMgr = get_super_module(table).ResourceManager
        except AttributeError:
            raise AttributeError('Backend does not have a ResourceManager class')

        param_grids = self.params['param_grid']

        if n_jobs is None:
            n_jobs = self.params['n_jobs']
        else:
            n_jobs = max(1, int(n_jobs))

        estimator = self.params['estimator']
        if type(estimator) is type:
            estimator = estimator()
        
        cv = self.params['cv']

        with ResMgr() as mgr:

            if isinstance(cv, numbers.Integral) or isinstance(cv, numbers.Real):
                cvtables = mgr.split_data(table, k=cv)
            else:
                cvtables = list(cv)

            parallel = mgr.is_parallelizable(*cvtables[0])

            if n_jobs > 1 and not parallel:
                warnings.warn('Either the current backend does not support parallel '
                              'execution or the data is not globally available; '
                              'The grid search will be done sequentially.',
                              RuntimeWarning) 

            if n_jobs > 1 and cvtables and parallel:
                params = [(mgr, estimator, train_table, score_table, parallel, params) 
                           for params in param_iter(param_grids)
                           for train_table, score_table in cvtables]
                pool = Pool(processes=n_jobs)
                out = pool.map(_fit_and_score, params)
                pool.close()
                pool.join()

            elif cvtables:
                out = []
                for params in param_iter(param_grids):
                    for train_table, score_table in cvtables:
                        out.append(_fit_and_score((mgr, estimator, train_table,
                                                   score_table, parallel, params)))
                out = [x for x in out if x is not None]

        n_fits = len(out)
        n_folds = len(cvtables)

        if not out:
            return

        grid_scores = []

        # If no score_type was specified, set it automatically
        score_type = self.params['score_type']
        if not score_type:
            if 'AverageSquaredError' in out[0].columns:
                score_type = 'AverageSquaredError'
            else:
                score_type = 'MisClassificationRate'

        for rowidx in list(out[0].index):
            for grid_start in range(0, n_fits, n_folds):
                n_test_samples = 0
                score = 0
                all_scores = []
                clock_time = 0

                for df in out[grid_start:grid_start + n_folds]:
                    this_score = df.ix[rowidx, score_type]
                    this_n_test_samples = df.ix[rowidx, 'NObsUsed']
                    this_parameters = df.ix[rowidx, 'FitParameters']

                    all_scores.append(this_score)

                    clock_time += df.ix[rowidx, 'ClockTime']
                
                    this_score *= this_n_test_samples
                    n_test_samples += this_n_test_samples
                    score += this_score
          
                score /= float(n_test_samples)
                clock_time /= float(n_test_samples)
           
                grid_scores.append((rowidx, score, np.std(all_scores),
                                    this_parameters, all_scores, clock_time))
        
        out = pd.DataFrame(grid_scores, 
                           columns=['Index', 'MeanScore', 'ScoreStd', 'Parameters',
                                    'FoldScores', 'MeanClockTime'],
                          ).set_index('Index').sort_values('MeanScore')
        if str(out.index.dtype) != 'object':
            out = out.reset_index(drop=True)
        out.index.name = None

        return out
