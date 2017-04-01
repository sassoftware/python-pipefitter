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

from . import _assess
from ....estimator import forest
from .base import EstimatorMixIn, ModelMixIn, ModelType, check_action


class DecisionForest(forest.DecisionForest, EstimatorMixIn):
    ''' Decision Forest for CAS '''

    def fit(self, table, **kwargs):
        ''' Fit function for random forest '''
        params = self.remap_params(type(self).static_params, kwargs)
        params['casout'] = self.create_model_table(table.get_connection(),
                                                   prefix='kmodelforest_')

        table.loadactionset('decisiontree', _apptag='UI', _messagelevel='error')

        return DecisionForestModel(params['casout'], kwargs,
                                   check_action(table.decisiontree.foresttrain(**params)),
                                   backend=self._get_backend(table))


class DecisionForestModel(forest.DecisionForestModel, ModelMixIn):
    ''' Decision Forest model for CAS '''

    def get_predicted_col_name(self, level_info):
        if self._model_type == ModelType.classification:
            return '_RF_P_', '_RF_LEVEL_', None
        else:
            return '_RF_PredMean_', None, None
    
    def get_default_event_level(self, n_levels):
        if n_levels == 2:
            return 0
        else:
            return n_levels - 1
            
    def score(self, table, event=None):
        ''' 
        Fit function for CAS

        Parameters
        ----------
        table : CASTable
            The CASTable object to score 

        Returns
        -------
        :class:`pandas.DataFrame`

        '''
        self._check_backend(table)
        self._model_type = self.get_model_type(table)
        table.loadactionset('decisiontree', _apptag='UI', _messagelevel='error')
        score_out = self.create_output_table(table.get_connection(), prefix='kscoreforest')
        if self._model_type == ModelType.classification:
            res = check_action(table.decisiontree.forestscore(modeltable=self.data, 
                                                              copyvars=[self.params['target']],
                                                              casout=score_out,
                                                              assess=True, encodename = True, 
                                                              assessonerow=True))
        else:
            res = check_action(table.decisiontree.forestscore(modeltable=self.data, 
                                                              copyvars=[self.params['target']],
                                                              casout=score_out))
        assess_res, assess_info = _assess.assess_model(self, score_out, event)
        return self.make_score_output(assess_res, assess_info)

    def unload(self):
        ''' Drop the model table '''
        if self.data is not None:
            self.data.table.droptable(_messagelevel='error', _apptag='UI')
