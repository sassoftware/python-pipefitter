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
Base classes for estimators

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import copy
import pandas as pd
import re
import six
import uuid
from ..transformer.utils import get_columns

PARAM_MAP = {
    'cf_level': 'cflev',
    'criterion': 'crit',
    'direct': {True: dict(arch='direct'), False: dict(arch='mlp')},
    'early_stop_stagnation': 'earlystop.stagnation',
    'max_branches': 'maxbranch',
    'max_depth': 'maxlevel',
    'n_trees': 'ntree',
    'out_of_bag': 'oob',
    'var_importance': 'varimp',

    'NeuralNetwork:annealing_rate': 'nloopts.sgdopt.annealingrate',
    'NeuralNetwork:optimization': 'nloopts.algorithm',
    'NeuralNetwork:ridge': 'nloopts.optmlopt.regl2',
    'NeuralNetwork:max_iters': 'nloopts.optmlopt.maxiters',
    'NeuralNetwork:max_time': 'nloopts.optmlopt.maxtime',
    'NeuralNetwork:num_tries': 'ntries',
    'NeuralNetwork:lasso': 'nloopts.optmlopt.regl2',
    'NeuralNetwork:learning_rate': 'nloopts.sgdopt.learningrate',

    'LogisticRegression:criterion': 'criterion',
    'LogisticRegression:selection': 'selection.method',
    'LogisticRegression:max_effects': 'selection.maxeffects',
    'LogisticRegression:intercept': {True: {'model.noint': False},
                                     False: {'model.noint': True}},

    'LinearRegression:criterion': 'criterion',
    'LinearRegression:selection': 'selection.method',
    'LinearRegression:max_effects': 'selection.maxeffects',
    'LinearRegression:intercept': {True: {'model.noint': False},
                                   False: {'model.noint': True}},
}


def check_action(res):
    ''' 
    Check the action status and raise expection if it failed

    Parameters
    ----------
    res : CASResults
        The results from the action

    Raises
    ------
    RuntimeError
        If the action failed

    Returns
    -------
    :class:`CASResults`

    '''
    if res.severity > 1:
        raise RuntimeError(res.status)
    return res

def create_output_table(self, conn, prefix='', caslib=None, replace=True):
    '''
    Create an output table definition

    Parameters
    ----------
    conn : CAS object
        The connection to use to create the output table
    prefix : string, optional
        A prefix to use at the beginning of the table name
    caslib : string, optional
        The CASLib to use for the table
    replace : boolean, optional
        Should the output table replace an existing table of the same name?

    Raises
    ------
    ValueError
        If the connection object is None

    Returns
    -------
    CASTable object

    '''
    if conn is None:
        raise ValueError('There is no connection object to create the output table.')

    name = ('%s%s' % (prefix, uuid.uuid4())).replace('-', '_')

    if caslib:
        return conn.CASTable(name, caslib=caslib, replace=replace)

    return conn.CASTable(name, replace=replace)


def merge_dicts(d1, d2):
    ''' Merge nested dictionaries '''
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        return d2
    d1 = copy.deepcopy(d1)
    for k in d2:
        if k in d1:
            d1[k] = merge_dicts(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    return d1


class EstimatorMixIn(object):
    ''' Additional Estimator methods '''

    def _expand_compound_keys(self, key, value):
        if '.' not in key:
            return {key: value}
        out = {}
        current = out
        keys = key.split('.')
        for i, key in enumerate(keys):
            if i == (len(keys) - 1):
                break
            current[key] = {}
            current = current[key]
        current[key] = value
        return out

    def remap_params(self, *params):
        '''
        Remap key names and values

        Parameters
        ----------
        *params : one or more dicts
           The dictionaries of parameters to remap

        Returns
        -------
        dict

        '''
        out = {}
        for item in params:
            for key, value in item.items():
                if isinstance(value, six.string_types):
                    value = value.replace('_', '')
                if self.__class__.__name__ + ':' + key in PARAM_MAP:
                    key = PARAM_MAP[self.__class__.__name__ + ':' + key]
                    if isinstance(key, dict):
                        out.update(self.remap_params(key[value]))
                        continue
                elif key in PARAM_MAP:
                    key = PARAM_MAP[key]
                    if isinstance(key, dict):
                        out.update(self.remap_params(key[value]))
                        continue
                else:
                    key = key.replace('_', '')
                out = merge_dicts(out, self._expand_compound_keys(key, value))
        # Make sure that nominals is a subset of inputs
        if out.get('inputs', None) and out.get('nominals', None):
            nominals = set(out['nominals'])
            new_nominals = []
            #make sure that the target is considered
            if out['target'] in nominals:
                new_nominals.append(out['target'])
            for item in out['inputs']:
                if item in nominals:
                    new_nominals.append(item)
            out['nominals'] = new_nominals
        return out

    def create_model_table(self, conn, prefix='', caslib=None, replace=True):
        '''
        Create an output table definition

        Parameters
        ----------
        conn : CAS object
            The connection to use to create the output table
        prefix : string, optional
            A prefix to use at the beginning of the table name
        caslib : string, optional
            The CASLib to use for the table
        replace : boolean, optional
            Should the output table replace an existing table of the same name?

        Raises
        ------
        ValueError
            If the connection object is None

        Returns
        -------
        CASTable object

        '''
        return create_output_table(self, conn, prefix=prefix,
                                   caslib=caslib, replace=replace)


class ModelType:
    classification = 0
    regression = 1

    
class ModelMixIn(object):
    ''' Additional Model methods '''

    def create_output_table(self, conn, prefix='', caslib=None, replace=True):
        '''
        Create an output table definition

        Parameters
        ----------
        conn : CAS object
            The connection to use to create the output table
        prefix : string, optional
            A prefix to use at the beginning of the table name
        caslib : string, optional
            The CASLib to use for the table
        replace : boolean, optional
            Should the output table replace an existing table of the same name?

        Raises
        ------
        ValueError
            If the connection object is None

        Returns
        -------
        CASTable object

        '''
        return create_output_table(self, conn, prefix=prefix,
                                   caslib=caslib, replace=replace)

    def make_score_output(self, assess_res, assess_info):
        out = pd.Series()
        out.loc['Target'] = assess_info['target']
        out.loc['Level'] = assess_info['level']
        out.loc['Var'] = assess_info['var']
        out.loc['NBins'] = assess_info['nbins']
        out.loc['NObsUsed'] = assess_info['nobsused']
        out.loc['TargetCount'] = assess_info['targetcount']
        out.loc['TargetMiss'] = assess_info['targetmiss']
        out.loc['PredCount'] = assess_info['predcount']
        out.loc['PredMiss'] = assess_info['predmiss']
        if 'event' in assess_info.keys():
            out.loc['Event'] = assess_info['event']
            out.loc['EventCount'] = assess_info['eventcount']
            out.loc['NonEventCount'] = assess_info['noneventcount']
            out.loc['EventMiss'] = assess_info['eventmiss']
        out = out.append(assess_res)
        return out
        
    def get_predicted_col_name(self, level_info):
        return None, None, None
        
    def get_model_type(self, table):
        '''
            Determine the model type, namely, classification vs regression.
            For numeric targets, classification if target in nominal, else
            regression.
            For non-numerics, always classification.
        '''
        if self.params['target'] in self.params['nominals']:
            return ModelType.classification
        else:
            columns, char_cols, num_cols = get_columns(table)
            for var in char_cols:
                if self.params['target'].strip().upper() == var.strip().upper():
                    return ModelType.classification
            return ModelType.regression
    
    def get_default_event_level(self, n_levels):
        '''
            Each model should override this to set its default event level. 
            This is so as to get matching defults between the CAS and SAS 
            back ends. e.g. hpneural first + desc, hplogis  first + asce etc
        '''
        return None
