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
CAS Regression Implementations

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import pandas as pd
from . import _assess
from ....estimator import regression
from .base import EstimatorMixIn, ModelMixIn, ModelType, check_action


def set_model_params(params):
    ''' Setup model parameters for various selection methods '''

    model = dict()
    if 'inputs' in params:
        model['effects'] = params.pop('inputs')
    if 'target' in params:
        model['depvars'] = params.pop('target')
    if 'nominals' in params:
        params['class'] = params.pop('nominals')
    params['model'] = model

    selection = params.get('selection', {})
    selection['details'] = 'summary'
    method = selection.get('method', None)

    sl = params.pop('siglevel', 0.05)
    criterion = params.pop('criterion', None)

    if method == 'fast_backward':
        selection['select'] = 'sl'
        selection['slstay'] = sl

    elif method == 'backward':
        selection['select'] = criterion or 'aic'
        selection['stop'] = criterion or 'aic'
        selection['choose'] = criterion or 'aic'

    elif method == 'forward':
        selection['details'] = 'all'
        selection['select'] = criterion or 'sbc'
        selection['stop'] = criterion or 'sbc'
        selection['choose'] = criterion or 'sbc'
        if selection['select'] == 'sl':
            selection['slentry'] = sl
            selection['choose'] = 'sbc'

    elif method == 'stepwise':
        selection['select'] = criterion or 'sl'
        selection['stop'] = criterion or 'sl'
        selection['choose'] = criterion or 'sbc'
        if selection['select'] == 'sl':
            selection['slentry'] = sl
            selection['slstay'] = sl
            selection['choose'] = 'sbc'

    elif method == 'lasso':
        selection['stop'] = criterion or 'aicc'
        selection['choose'] = criterion or 'aicc'

    elif method == 'adaptive_lasso':
        selection['stop'] = criterion or 'aic'
        selection['choose'] = criterion or 'aic'
        selection['adaptive'] = True

    return params


class LogisticRegression(regression.LogisticRegression, EstimatorMixIn):
    ''' Logistic Regression for CAS '''

    def fit(self, table, **kwargs):
        ''' Fit function for logistic regression '''
        params = self.remap_params(type(self).static_params, kwargs)
        params = set_model_params(params)
        
        table.loadactionset('regression', _apptag='UI', _messagelevel='error')

        results = check_action(table.regression.logistic(code=dict(), **params))

        model = dict(dscode='\n'.join(results.pop('_code_')['SASCode'].tolist()))

        return LogisticRegressionModel(model, kwargs, results,
                                       backend=self._get_backend(table))


class LogisticRegressionModel(regression.LogisticRegressionModel, ModelMixIn):
    ''' Logistic Regression model for CAS '''
    
    def get_predicted_col_name(self, level_info):
        if self._model_type == ModelType.classification:
            if level_info['nlevels'] == 2:
                if level_info['eventindex'] == 0:
                    return 'P_' + self.params['target'], None, None
                else:
                    #logistic models the prob of lowest level
                    var = '_P_' + self.params['target'] + '_'
                    pred_p = 'P_' + self.params['target']
                    cmp_pgm = '1.0 - ' + pred_p
                    return var, None, cmp_pgm
            else:
                #a la logistic regression (for y with levels (a, b, c), 
                #return P_yc
                event_index = level_info['eventindex']
                var = 'P_' + self.params['target'] + \
                      level_info['levels'][event_index].strip()
                cmp_pgm = '1.0/ (1.0 + '
                for i in range(0, level_info['nlevels'] -1):
                    temp = 'P_' + self.params['target'] + \
                           level_info['levels'][i].strip()
                    cmp_pgm += temp
                    if i < level_info['nlevels'] - 2:
                        cmp_pgm += ' + '
                cmp_pgm += ")"
                if level_info['eventindex'] != level_info['nlevels'] - 1:
                    cmp_pgm = var + " * " + cmp_pgm
                var = "_" + var
                return var, None, cmp_pgm
        else:
            return None, None, None
            
            
    def get_default_event_level(self, n_levels):
        if n_levels == 2:
            return 0
        else:
            return n_levels - 1
            
    def score(self, table, event=None):
        '''
        Score function for logistic regression

        Parameters
        ----------
        table : CASTable
            The CASTable to score

        Returns
        -------
        :class:`pandas.DataFrame`

        '''
        self._check_backend(table)
        self._model_type = self.get_model_type(table)
        self._model_name = "logistic"
        score_out = table.datastep(self.data['dscode'])
        assess_res, assess_info = _assess.assess_model(self, score_out, event)
        return self.make_score_output(assess_res, assess_info)

    def unload(self):
        ''' Drop the model table '''
        pass


class LinearRegression(regression.LinearRegression, EstimatorMixIn):
    ''' Linear Regression for CAS '''

    def fit(self, table, **kwargs):
        ''' Fit function for logistic regression '''
        params = self.remap_params(type(self).static_params, kwargs)
        params = set_model_params(params)

        table.loadactionset('regression', _apptag='UI', _messagelevel='error')

        results = check_action(table.regression.glm(code=dict(), **params))

        model = dict(dscode='\n'.join(results.pop('_code_')['SASCode'].tolist()))

        return LinearRegressionModel(model, kwargs, results,
                                       backend=self._get_backend(table))


class LinearRegressionModel(regression.LinearRegressionModel, ModelMixIn):
    ''' Linear Regression model for CAS '''

    def get_predicted_col_name(self, level_info):
        if self._model_type == ModelType.classification:
            None, None, None
        else:
            return "P_"+self.params['target'], None, None
            
    def score(self, table):
        '''
        Score function for linear regression

        Parameters
        ----------
        table : CASTable
            The CASTable to score

        Returns
        -------
        :class:`pandas.DataFrame`

        '''
        self._check_backend(table)
        self._model_type = self.get_model_type(table)
        score_out = table.datastep(self.data['dscode'])
        assess_res, assess_info = _assess.assess_model(self, score_out, None)
        return self.make_score_output(assess_res, assess_info)

    def unload(self):
        ''' Drop the model table '''
        pass
