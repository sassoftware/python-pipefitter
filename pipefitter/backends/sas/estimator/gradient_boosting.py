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

from ....estimator import gradient_boosting as gb
from .base import EstimatorMixIn, ModelMixIn


class GBTree(gb.GBTree, EstimatorMixIn):
    ''' Gradient Boosting Tree for SAS '''

    def fit(self, table, **kwargs):
        ''' Fit function for decision tree '''
        params = kwargs.copy()
        params.update(type(self).static_params)
        params = self.remap_params('GB', params)
        model  = self.create_model_table(table.sas)
        name     = model.get('name')
        codename = model.get('path') + name
        if isinstance(params['procopts'], dict):
            n_bins = params['procopts'].pop('n_bins', None)
            if n_bins:
                params['procopts']['intervalbins'] = n_bins
                params['procopts']['categoricalbins'] = n_bins
            procoptsStr = ' '.join('{}={}'.format(key, val) for key, val in params['procopts'].items())
            params['procopts'] = procoptsStr
        

        self.ml = table.sas.sasml()

        return GBTreeModel(model, kwargs,
                   self.ml.treeboost(data=table, target=kwargs['target'], save=True, **params),
                   backend=self._get_backend(table))


class GBTreeModel(gb.GBTreeModel, ModelMixIn):
    ''' Gradient Boosting Tree model for SAS '''

    def score(self, table):
        ''' Score function for decision tree '''
        self._check_backend(table)
        df = self.commonScore(table, algo='GBTree')
        return df
