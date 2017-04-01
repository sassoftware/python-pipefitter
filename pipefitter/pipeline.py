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

''' Pipeline object '''

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import pandas as pd
import re
import six
from collections import Sequence
from .base import BaseTransformer, BaseEstimator, BaseModel

def tosequence(obj):
    ''' Cast an iterable to a sequence '''
    if isinstance(obj, np.ndarray):
        return np.asarray(obj)
    elif isinstance(obj, Sequence):
        return obj
    return list(obj)


@six.python_2_unicode_compatible
class Pipeline(object):
    '''
    Execute a series of transformers and estimators

    Parameters
    ----------
    stages : one or more transformers/estimators
        The stages of the pipeline to execute

    Examples
    --------
    
    Basic pipeline of imputers and an estimator:

    >>> mean_imp = Imputer(Imputer.MEAN)
    >>> mode_imp = Imputer(Imputer.MODE)
    >>> dtree = DecisionTree(target='Origin',
    ...                      nominals=['Type', 'Cylinders', 'Origin'],
    ...                      inputs=['MPG_City', 'MPG_Highway', 'Length',
    ...                              'Weight', 'Type', 'Cylinders'])
    >>> pipe = Pipeline([mean_imp, mode_imp, dtree])

    Returns
    -------
    :class:`Pipeline`

    '''

    def __init__(self, stages):
        self.stages = tosequence(stages)
        self._extra_params = []
        for item in self.stages:
            if not isinstance(item, BaseTransformer):
                raise TypeError('%s is not a transformer or estimator' % item)

    def __str__(self):
        return '%s([%s])' % (type(self).__name__,
                             ', '.join(str(x) for x in self.stages))

    def __repr__(self):
        return str(self)

    def set_params(self, *args, **kwargs):
        '''
        Set additional parameters for the estimators in the pipeline

        Parameters
        ----------
        *args : positional parameters, optional
            Any valid parameters to the estimators' ``fit`` method
        **kwargs : keyword parameters, optional
            Any valid keyword parameters to the estimators' ``fit`` method

        '''
        self._extra_params.extend(list(args))
        self._extra_params.append(kwargs)

    def fit(self, table, *args, **kwargs):
        '''
        Train the models using the stages in the pipeline

        Notes
        -----
        Parameters passed in on this method are not persisted on 
        the pipeline.  They are only used during the scope of this method.

        Parameters
        ----------
        table : data set
            Any data set object supported by the transformers and
            estimators in the pipeline stages
        *args : positional parameters, optional
            Any valid parameters to the estimators' ``fit`` method
        **kwargs : keyword parameters, optional
            Any valid keyword parameters to the estimators' ``fit`` method

        Examples
        --------

        Basic pipeline fit using imputers and an estimator:

        >>> mean_imp = Imputer(Imputer.MEAN)
        >>> mode_imp = Imputer(Imputer.MODE)
        >>> dtree = DecisionTree(target='Origin',
        ...                      nominals=['Type', 'Cylinders', 'Origin'],
        ...                      inputs=['MPG_City', 'MPG_Highway', 'Length',
        ...                              'Weight', 'Type', 'Cylinders'])
        >>> pipe = Pipeline([mean_imp, mode_imp, dtree])
        >>> model = pipe.fit(data)

        Returns
        -------
        :class:`PipelineModel`

        '''
        out = []
        last_idx = len(self.stages) - 1

        extra_params = list(self._extra_params)
        extra_params.extend(args)
        extra_params.append(kwargs)

        for i, stage in enumerate(self.stages):
            params = stage.get_filtered_params(*extra_params)
            if isinstance(stage, BaseEstimator):
                out.append(stage.fit(table, **params))
                if i == last_idx:
                    break
            else:
                out.append(stage)
            table = out[-1].transform(table)

        if out:
            return PipelineModel(out)

    def transform(self, table, *args, **kwargs):
        '''
        Execute the transformations in this pipeline only

        Parameters
        ----------
        table : data set
            Any data set object supported by the transformers and
            estimators in the pipeline stages
        *args : positional parameters, optional
            Any valid parameters to the transformers' ``transform`` method
        **kwargs : keyword parameters, optional
            Any valid keyword parameters to the transformers' ``transform`` method 

        Notes
        -----
        When the pipeline contains estimators, they typically just pass the
        input table on to the next stage of the pipeline.

        Examples
        --------

        Basic pipeline fit using imputers and an estimator:

        >>> mean_imp = Imputer(Imputer.MEAN)
        >>> mode_imp = Imputer(Imputer.MODE)
        >>> dtree = DecisionTree(target='Origin',
        ...                      nominals=['Type', 'Cylinders', 'Origin'],
        ...                      inputs=['MPG_City', 'MPG_Highway', 'Length',
        ...                              'Weight', 'Type', 'Cylinders'])
        >>> pipe = Pipeline([mean_imp, mode_imp, dtree])
        >>> new_table = pipe.transform(data)

        Returns
        -------
        data set
            The same type of data set as passed in `table`

        '''
        out = []
        last_idx = len(self.stages) - 1

        extra_params = list(self._extra_params)
        extra_params.extend(args)
        extra_params.append(kwargs)

        for i, stage in enumerate(self.stages):
            params = stage.get_filtered_params(*extra_params)
            if isinstance(stage, BaseEstimator):
                out.append(stage.fit(table, **params))
                if i == last_idx:
                    break
            else:
                out.append(stage)
            table = out[-1].transform(table)

        return table

    def __getitem__(self, idx):
        return self.stages[idx]


@six.python_2_unicode_compatible
class PipelineModel(object):
    ''' 
    Trained model for a Pipeline 

    Notes
    -----
    This object is not instantiated directly.  It is the result of
    calling the ``fit`` method of the :class:`Pipeline` object.

    Parameters
    ----------
    stages : list of transformors / models
        A list of the elements of the fitted Pipeline.

    Returns
    -------
    :class:`PipelineModel`

    '''

    def __init__(self, stages):
        self.stages = tosequence(stages)

    def __str__(self):
        return '%s([%s])' % (type(self).__name__,
                             ', '.join(str(x) for x in self.stages))

    def __repr__(self):
        return str(self)

    def score(self, table, **kwargs):
        '''
        Apply transformations and score the data using the trained model

        Parameters
        ----------
        table : data set
            A data set that is of the same type as the training data set

        Examples
        --------

        Basic pipeline model transform using imputers and an estimator:

        >>> mean_imp = Imputer(Imputer.MEAN)
        >>> mode_imp = Imputer(Imputer.MODE)
        >>> dtree = DecisionTree(target='Origin',
        ...                      nominals=['Type', 'Cylinders', 'Origin'],
        ...                      inputs=['MPG_City', 'MPG_Highway', 'Length',
        ...                              'Weight', 'Type', 'Cylinders'])
        >>> pipe = Pipeline([mean_imp, mode_imp, dtree])
        >>> model = pipe.fit(training_data)
        >>> score = model.score(data)

        Returns
        -------
        :class:`pandas.DataFrame`

        '''
        scores = []
        names = {}

        for i, stage in enumerate(self.stages):
            if isinstance(stage, BaseModel):
                scores.append(stage.score(table, **kwargs))
                name = re.sub(r'Model$', '', type(stage).__name__)
                if name in names:
                    names[name] += 1
                    name = '%s%s' % (name, names[name])
                else:
                    names[name] = 0
                scores[-1].name = name
            table = stage.transform(table)

        if scores:
            if len(scores) == 1:
                return scores[0]
            return pd.DataFrame(scores)

    def transform(self, table):
        '''
        Run the transforms in the trained pipeline

        Parameters
        ----------
        table : data set
            A data set that is of the same type as the training data set

        Examples
        --------

        Basic pipeline model transform using imputers and an estimator:

        >>> mean_imp = Imputer(Imputer.MEAN)
        >>> mode_imp = Imputer(Imputer.MODE)
        >>> dtree = DecisionTree(target='Origin',
        ...                      nominals=['Type', 'Cylinders', 'Origin'],
        ...                      inputs=['MPG_City', 'MPG_Highway', 'Length',
        ...                              'Weight', 'Type', 'Cylinders'])
        >>> pipe = Pipeline([mean_imp, mode_imp, dtree])
        >>> model = pipe.fit(training_data)
        >>> new_table = model.transform(data)

        Returns
        -------
        data set
            A data set of the same type that was passed in `table`

        '''
        for stage in self.stages:
            table = stage.transform(table)
        return table

    def __getitem__(self, idx):
        return self.stages[idx]

    def unload(self):
        ''' Unload model resources '''
        for stage in self.stages:
            if isinstance(stage, BaseModel):
                stage.unload()
