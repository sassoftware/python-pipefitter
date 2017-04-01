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

from ....estimator import tree
from .base import EstimatorMixIn, ModelMixIn
# from pdb import set_trace as bp


class DecisionTree(tree.DecisionTree, EstimatorMixIn):
    ''' Decision Tree for SAS '''

    def fit(self, table, **kwargs):
        ''' Fit function for decision tree '''
        params = kwargs.copy()
        params.update(type(self).static_params)
        params = self.remap_params('DT', params)
        model  = self.create_model_table(table.sas)
        codename = model.get('path') + model.get('name')

        # use data mining style for missing to keep more data
        try:
            if len(params['procopts']) > 0:
                params['procopts'] += ' assignmissing = similar '
            else:
                params['procopts'] = 'assignmissing = similar'
        except:
            pass
        pruneKey = params.pop('prune', None)
        if pruneKey is not None:
            if pruneKey == True:
                pass
                #params['prune'] = 'c45' # if the target is binary, nominal
                #params['prune'] = 'ase' # if the target is interval
            else:
                params['prune'] = 'off'
        stat = table.sas.sasstat()

        return DecisionTreeModel(model, kwargs,
                   stat.hpsplit(data=table, target=kwargs['target'], code=codename, **params),
                   backend=self._get_backend(table))


class DecisionTreeModel(tree.DecisionTreeModel, ModelMixIn):
    ''' Decision Tree trained model for SAS '''

    def score(self, table):
        ''' Score function for decision tree '''
        self._check_backend(table)
        df = self.commonScore(table, algo='DecisionTree')
        return df

