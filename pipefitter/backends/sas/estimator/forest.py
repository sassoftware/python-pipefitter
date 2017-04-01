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
Decision Forest

'''

from __future__ import print_function, division, absolute_import, unicode_literals

from ....estimator import forest
from .base import EstimatorMixIn, ModelMixIn

class DecisionForest(forest.DecisionForest, EstimatorMixIn):
    ''' Decision Forest for SAS '''

    def fit(self, table, **kwargs):
        ''' Fit function for random forest '''
        params = kwargs.copy()
        params.update(type(self).static_params)
        params = self.remap_params('RF', params)
        model  = self.create_model_table(table.sas)
        codename = model.get('path') + model.get('name')

        ml = table.sas.sasml()

        return DecisionForestModel(model, kwargs,
                   ml.forest(data=table, target=kwargs['target'], save=codename, **params),
                   backend=self._get_backend(table))


class DecisionForestModel(forest.DecisionForestModel, ModelMixIn):
    ''' Decision Forest model for SAS '''

    def score(self, table):
        ''' Score function for random forest '''
        self._check_backend(table)
        df = self.commonScore(table, algo='DecisionForest')
        return df
