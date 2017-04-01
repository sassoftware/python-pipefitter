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
Neural Network

'''
from __future__ import print_function, division, absolute_import, unicode_literals

from ....estimator import neural_net
from .base import EstimatorMixIn, ModelMixIn
# from pdb import set_trace as bp

class NeuralNetwork(neural_net.NeuralNetwork, EstimatorMixIn):
    ''' Neural Network for SAS '''

    def fit(self, table, **kwargs):
        ''' Fit function for decision tree '''
        params = kwargs.copy()
        params.update(type(self).static_params)
        params = self.remap_params('NN', params)
        model  = self.create_model_table(table.sas)
        codename = model.get('path') + model.get('name')
        hidden = params.pop('hidden', None)
        direct = kwargs.pop('direct', None)
        act = kwargs.pop('acts', None)
        t_act = params.pop('target_act', None) 
        try:
            if act.casefold() == 'logistic':
                direct = None
                hidden = None
                t_act  = None
                params['architecture'] = 'Logistic'
        except:
            pass
        
        if hidden is not None:
            if act is not None:
                if act == 'softplus':
                    act = 'softsign'
                params['hidden'] = '%s / act=%s' %(' '.join(str(x) for x in hidden), act)

        if direct is not None:
            if direct:
                params['architecture'] = 'MLP Direct'
            else:
                params['architecture'] = 'MLP '
        if t_act is not None:
            params['targOpts'] = {}
            params['targOpts']['act'] = t_act
            if t_act == 'exp':
                 params['targOpts']['error'] = 'gamma'

        maxiter = kwargs.pop('max_iters', None)
        numtries = params.pop('numtries', None)
        if maxiter is not None:
            params['train'] = 'maxiter = %s' % maxiter 
        if numtries is not None:
            if len(params['train']) == 0:
                params['train'] = ' numtries = %s ' % numtries 
            else:
                params['train'] += ' numtries = %s ' % numtries
        ml = table.sas.sasml()
        return NeuralNetworkModel(model, kwargs,
                   ml.neural(data=table, target=kwargs['target'], code=codename, **params),
                   backend=self._get_backend(table))


class NeuralNetworkModel(neural_net.NeuralNetworkModel, ModelMixIn):
    ''' Neural Network trained model for SAS '''

    def score(self, table):
        ''' Score function for Neural Network '''
        self._check_backend(table)
        df = self.commonScore(table, algo='NeuralNetwork')
        return df
