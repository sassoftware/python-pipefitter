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

from ....estimator import regression
from .base import EstimatorMixIn, ModelMixIn
##  from pdb import set_trace as bp

class LogisticRegression(regression.LogisticRegression, EstimatorMixIn):
    ''' Logistic Regression for SAS '''

    def fit(self, table, **kwargs):
        ''' Fit function for Logistic Regression'''
        params = kwargs.copy()
        params.update(type(self).static_params)
        params = self.remap_params('REG', params)
        model  = self.create_model_table(table.sas)
        codename = model.get('path') + model.get('name')
        method = params.pop('selection', {})
        sl = params.pop('sig_level', 0.05)
        criterion = params.pop('criterion', None)

        selection = {}
        if method == "backward":
            selection['method'] = 'backward'
            selection['select'] = criterion or 'aic'
            selection['stop'] = criterion or 'aic'
            selection['choose'] = criterion or 'aic'

        elif method == "forward":
            selection['method'] = 'forward'
            selection['details'] = 'all'
            selection['select'] = criterion or 'sbc'
            selection['stop'] = criterion or 'sbc'
            selection['choose'] = criterion or 'sbc'
            if selection['select'] == 'sl':
                selection['choose'] = 'sbc'

        elif method == "stepwise":
            selection['method'] = 'stepwise'
            selection['select'] = criterion or 'sl'
            selection['stop'] = criterion or 'sl'
            selection['choose'] = criterion or 'sbc'
            if selection['select'] == 'sl':
                selection['slentry'] = sl
                selection['slstay'] = sl
                selection['choose'] = 'sbc'

        try:
            if params['maxeffects'] > 0 and selection['method'] != 'backward':
                selection['maxeffects'] = params['maxeffects']
        except:
            pass

        params['selection'] = selection
        stat = table.sas.sasstat()
        intercept = ''
        if kwargs['intercept'] == False:
            intercept = '/ noint'
        modelstmt = str(kwargs['target'] + "=" + " ".join(kwargs['inputs']) + intercept)
        return LogisticRegressionModel(model, kwargs,
                                       stat.hplogistic(data=table, cls=kwargs['nominals'],
                                                       code=codename, model=modelstmt, 
                                                       **params),
                                       backend=self._get_backend(table))


class LogisticRegressionModel(regression.LogisticRegressionModel, ModelMixIn):
    ''' Regression trained model for SAS '''

    def score(self, table):
        ''' Score function for Regression '''
        self._check_backend(table)
        df = self.commonScore(table, algo='LogisticRegression')
        return df

class LinearRegression(regression.LinearRegression, EstimatorMixIn):
    ''' Linear Regression for SAS '''

    def fit(self, table, **kwargs):
        ''' Fit function for decision tree '''
        params = kwargs.copy()
        params.update(type(self).static_params)
        params = self.remap_params('REG', params)
        model  = self.create_model_table(table.sas)
        codename = model.get('path') + model.get('name')

        method = params.pop('selection', {})
        sl = params.pop('sig_level', None)
        criterion = params.pop('criterion', None)

        selection = {}
        if method == "backward":
            selection['method'] = 'backward'
            selection['select'] = criterion or 'aic'
            selection['stop'] = criterion or 'aic'
            selection['choose'] = criterion or 'aic'

        elif method == "forward":
            selection['method'] = 'forward'
            selection['details'] = 'all'
            selection['select'] = criterion or 'sbc'
            selection['stop'] = criterion or 'sbc'
            selection['choose'] = criterion or 'sbc'

        elif method == 'lasso':
            selection['method'] = 'lasso'
            selection['stop'] = criterion or 'aicc'
            selection['choose'] = criterion or 'aicc'

        elif method == "stepwise":
            selection['method'] = 'stepwise'
            selection['select'] = criterion or 'sl'
            selection['stop'] = criterion or 'sl'
            selection['choose'] = criterion or 'sbc'
            if sl is not None:
                selection['slentry'] = sl
                selection['slstay'] = sl
        try:
            if params['maxeffects'] > 0:
                selection['maxeffects'] = params['maxeffects']
        except:
            pass
        if 'method' in selection:
            params['selection'] = selection
      
        stat = table.sas.sasstat()
        intercept = ''
        if params['intercept'] == False:
            intercept = '/ noint'
        modelstmt = str(kwargs['target'] + "=" + " ".join(kwargs['inputs']) + intercept)
        return LinearRegressionModel(model, kwargs,
                                     stat.hpreg(data=table, cls=kwargs['nominals'],
                                                code=codename, model=modelstmt, **params),
                                     backend=self._get_backend(table))

class LinearRegressionModel(regression.LinearRegressionModel, ModelMixIn):
    ''' Linear Regression model for SAS '''

    def score(self, table):
        """

        :param table:
        :return:
        """
        self._check_backend(table)
        df = self.commonScore(table, algo='LinearRegression')
        return df
