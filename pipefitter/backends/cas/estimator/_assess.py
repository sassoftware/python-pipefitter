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
#  distributed under the License is distributed on an 'AS IS' BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from __future__ import print_function, division, absolute_import, unicode_literals

import collections
import numpy
import pandas as pd
from .base import ModelType, check_action

#
# A model information class
# 
# Parameters
# ----------
# type : ModelType
#     The model type either ModelType.classification, ModelType.regression
# target : string
#     The target variable
# predicted : string
#     The predicted variable
# 
# Returns
# --------
# :class: ``ModelInfo``
# example: ModelInfo(ModelType.classification, 'y', 'Pr_y')
#
_model_info_props = ['type', 'target', 'predicted']
ModelInfo = collections.namedtuple('ModelInfo', _model_info_props)


class ModelMetrics(object):
    '''
    ModelMetrics

    Parameters:
    -----------
    model_info : ModelInfo
        A ModelInfo object for basic model/prediction information
    assess_res : CASResult
        The return CASResult object from the assess action. 
        For classification, assess_res must be a dictionary with
        'LIFTInfo and ROCInfo' as keys, and for regression, it
        must be a dictionary with 'FitStat' as key.

    Returns
    -------
    :class:`ModelMetrics`

    ''' 
    def __init__(self, model_info, assess_res):
        self._validate_params(model_info, assess_res)
        self.model_info = model_info
        self.assess_res = assess_res

    def get_metrics(self):
        '''
        Get the model_metrics.

        Returns
        -------
        ClassMetrics or RegMetrics depending on the model type

        '''
        if self.model_info.type == ModelType.classification:
            return self._classification_metrics()
        else:
            return self._regression_metrics()

    def get_roc(self):
        ''' Get the (sensitivity, one_minus_specificity) values for ROC curve '''
        if self.model_info.type != ModelType.classification:
            raise ValueError('get_roc is available only for regression')
        res = {}
        res['sensitivity'] = self.assess_res['ROCInfo']['Sensitivity']
        res['one_minus_specificity'] = 1.0 - self.assess_res['ROCInfo']['Specificity']
        return res

    def get_lift_cumulative(self):
        ''' Get the (depth, lift_cumulative) values for cumulative lift chart '''
        if self.model_info.type != ModelType.classification:
            raise ValueError('get_lift_cumulative is available only for regression')
        res = {}
        res['depth'] = self.assess_res['LIFTInfo']['Depth']
        res['lift_cumulative'] = self.assess_res['LIFTInfo']['CumLift']
        return res

    def get_lift_best(self):
        ''' Get the (depth, lift_best) values for best lift chart '''
        if self.model_info.type != ModelType.classification:
            raise ValueError('get_lift_best is available only for regression')
        res = {}
        res['depth'] = self.assess_res['LIFTInfo']['Depth']
        res['lift_best'] = self.assess_res['LIFTInfo']['CumLiftBest']
        return res

    def get_gain(self):
        ''' Get the (depth, gain) values for gain chart '''
        if self.model_info.type != ModelType.classification:
            raise ValueError('get_gain is available only for regression')
        res = {}
        res['depth'] = self.assess_res['LIFTInfo']['Depth']
        res['gain'] = self.assess_res['LIFTInfo']['Gain']
        return res

    def get_gain_best(self):
        ''' Get the (depth, gain_best) values for best-gain chart '''
        if self.model_info.type != ModelType.classification:
            raise ValueError('get_gain_best is available only for regression')
        res = {}
        res['depth'] = self.assess_res['LIFTInfo']['Depth']
        res['gain_best'] = self.assess_res['LIFTInfo']['GainBest']
        return res

    def _classification_metrics(self):
        # extract info for metrics computation
        ks2_array = self.assess_res['ROCInfo']['KS2']
        ks_index = numpy.argmax(abs(ks2_array))
        cr_array = self.assess_res['ROCInfo']['ACC']
        cr_index = numpy.argmax(abs(cr_array))
        prob_array = self.assess_res['ROCInfo']['CutOff']
        roc = self.get_roc()
        # compute the classification metrics
        auc = ModelMetrics.compute_auc(roc)
        ks = abs(ks2_array[ks_index])
        ksr = 100 * ks
        kscut = prob_array[ks_index]
        cr = abs(cr_array[cr_index])
        crcut = prob_array[cr_index]
        #metrics = ClassMetrics(auc, ks, ksr, kscut, cr, crcut)
        metrics = pd.Series(dict(AreaUnderROCCurve=auc, KS=ks, KSCutOff=kscut,
                                 MisClassificationRate=(1-cr)*100, CRCut=crcut,
                                 ClassificationCutOff=0.5), dtype=object)
        return metrics

    def _regression_metrics(self):
        # extract the regression metrics
        fit_stats = self.assess_res['FitStat']
        metrics = pd.Series(dict(AverageSquaredError=fit_stats['ASE'][0],
                                 RootAverageSquaredError=fit_stats['RASE'][0],
                                 AverageAbsoluteError=fit_stats['MAE'][0],
                                 RootAverageAbsoluteError=fit_stats['RMAE'][0],
                                 AverageSquaredLogarithmicError=fit_stats['MSLE'][0],
                                 RootAverageSquaredLogarithmicError=fit_stats['RMSLE'][0]), 
                                 dtype=object)
        return metrics

    @staticmethod
    def compute_auc(roc):
        ''' Compute auc using trapezoidal integration '''
        # trapz returns negative if the arrays are in descending order
        # with respect to the x-axis.
        return abs(numpy.trapz(roc["sensitivity"], x = roc["one_minus_specificity"]))
        
    def _validate_params(self, model_info, assess_res):
        if not model_info:
            raise ValueError('model information required')
        if not assess_res:
            raise ValueError('assess action result required')
        if model_info.type not in (ModelType.regression, ModelType.classification):
            raise ValueError('invalid model type')
        if not (model_info.target and model_info.predicted):
            raise ValueError('unspecified target and predicted columns')
        if model_info.type == ModelType.classification:
            if 'LIFTInfo' not in assess_res.keys() or \
               'ROCInfo' not in assess_res.keys():
                raise ValueError('you need to specify event for classification model')
            if model_info.predicted.lower() != assess_res['LIFTInfo']['Variable'][0].lower():
                raise ValueError('invalid assess action result for classification')
        else:
            if 'FitStat' not in assess_res.keys():
                raise ValueError('invalid assess action result for regression')
        
    def __str__(self):
        out = []
        if self.model_info.type == ModelType.classification:
            out.append('model_type=classification')
        else:
            out.append('model_type=regression')
        out.append('target=%s' % self.model_info.target)
        out.append('predicted=%s' % self.model_info.predicted)
        return '%s(%s)' % (type(self).__name__, ', '.join(out))
        
    def __repr__(self):
        return str(self)
        

def get_target_level_info(model, score_out, event):
    '''
        Get target level information from the score table. 
        The assumption is all target levels occur in the 
        table to be scored. More sensible to extract it from
        the fit table.
    '''
    level_info = {}
    if model._model_type == ModelType.classification:
        res = check_action(score_out.simple.freq(inputs=[model.params['target']]))
        res = res['Frequency']
        fmt_vals = res['FmtVar']
        freq_vals = res['Frequency']
        n_levels = len(fmt_vals)
        if fmt_vals[0].strip() == ".":
            #exclude miss in level count
            n_levels -= 1
            level_info['misslevel'] = True
            level_info['levels'] = fmt_vals[1:]
        else:
            level_info['misslevel'] = False
            level_info['levels'] = fmt_vals 
        level_info['nlevels'] = n_levels
        if not event:
            event_index = model.get_default_event_level(n_levels)
        else:
            event_index = -1
            for i in xrange(n_levels):
                if level_info['levels'][i].strip().upper() == event.strip().upper():
                    event_index = i
                    break
            if event_index == -1:
                raise ValueError('invalid event specified')
        level_info['eventindex'] = event_index
        level_info['event'] = level_info['levels'][level_info['eventindex']]
        if level_info['misslevel']:
            level_info['frequency'] = freq_vals[1:]
        else:
            level_info['frequency'] = freq_vals
    else:
        level_info['nlevels'] = -1
        level_info['misslevel'] = False
        level_info['levels'] = None
        level_info['eventindex'] = -1
        level_info['event'] = None
        level_info['frequency'] = None
    return level_info


def make_assess_info(model, score_out, assess_input, level_info):
    '''
        collect assess related stuffs.
    '''
    assess_info = {}
    
    input_vars = [model.params['target'], assess_input]
    target_row = 0
    pred_row = 1
    res = check_action(score_out.dataPreprocess.highCardinality(inputs=input_vars))
    res = res['HighCardinalityDetails']
    div = 1
    if level_info['nlevels'] > 0:
        if hasattr(model, '_model_name') and model._model_name == "logistic":
            div = 1
        else:
            div = level_info['nlevels']
            if level_info['misslevel']:
                div += 1
    assess_info['targetcount'] = res['N'][target_row]/div
    assess_info['targetmiss'] = res['NMiss'][target_row]/div
    assess_info['predcount'] = res['N'][pred_row]/div
    assess_info['predmiss'] = res['NMiss'][pred_row]/div
    assess_info['target'] = model.params['target']
    assess_info['var'] = assess_input
    if model._model_type == ModelType.classification:
        assess_info["level"] = "CLASS"
    else:
        assess_info["level"] = "INTERVAL"
    assess_info["nobsused"] = assess_info['targetcount'] + assess_info['targetmiss']
    
    if model._model_type == ModelType.classification:
        n_read = assess_info['targetcount'] + assess_info['targetmiss']
        event_index = level_info['eventindex']
        assess_info['event'] = level_info['event']
        assess_info['eventcount'] = level_info['frequency'][event_index]/div
        assess_info['noneventcount'] = n_read - assess_info['eventcount'] - \
                                       assess_info['targetmiss']
        assess_info['eventmiss'] = assess_info['targetmiss']
    return assess_info

def check_assess_res(assess_res):
    '''
        Check whether there is a valid assess_result object to work with.
    '''
    if not assess_res:
        raise ValueError('model assessment failed')
    if('LIFTInfo' not in assess_res.keys()) and \
      ('ROCInfo' not in assess_res.keys()) and \
      ('FitStat' not in assess_res.keys()):
        raise ValueError('assess action failed')
    if 'LIFTInfo' in assess_res.keys():
        is_empty = assess_res['LIFTInfo'].empty
    if 'ROCInfo' in assess_res.keys():
        is_empty = assess_res['ROCInfo'].empty
    if 'FitStat' in assess_res.keys():
        is_empty = assess_res['FitStat'].empty
    if is_empty:
        raise ValueError('assess action failed')

def assess_model(model, score_out, event, drop_table=True):
    '''
    Call assess action for model assessment

    Parameters:
    -----------
    model : BaseEstimator
        The estimator object, already fit, to be scored/assessed.
    score_out: CASTable
        The castable object to score
    drop_table: bool
        To drop or not the scored table

    Returns
    -------
    :class:`ModelMetrics`

    '''
    # assess parameters
    n_bins = 100
    epsilon = 1e-6
    cut_step = 0.01
    max_iter = 1000 
    assess_params = {"nbins":n_bins, "epsilon":epsilon, "cutstep": cut_step, 
                    "maxiter":max_iter}
    
    level_info = get_target_level_info(model, score_out, event)
    assess_input, presponse, cmp_pgm = model.get_predicted_col_name(level_info)
    
    if cmp_pgm:
        score_out.append_computed_columns([assess_input], 
                                          '%s = %s' % (assess_input, cmp_pgm))
    assess_info = make_assess_info(model, score_out, assess_input, level_info)
    assess_info.update(assess_params)
    
    assess_input = assess_info['var']
    target = assess_info['target']
    if model._model_type == ModelType.classification:
        if not event:
            event = assess_info['event']
        assess_res = check_action(score_out.percentile.assess(inputs=[assess_input],
                                                              presponse = presponse,
                                                              event=event,
                                                              response=target,
                                                              **assess_params))
        model_info = ModelInfo(ModelType.classification, target, assess_input)
    else:
        assess_res = check_action(score_out.percentile.assess(inputs=[assess_input], 
                                                              response=target,
                                                              **assess_params))
        model_info = ModelInfo(ModelType.regression, target, assess_input)
    check_assess_res(assess_res)
    metrics = ModelMetrics(model_info, assess_res)
    if drop_table:
        score_out.droptable(_messagelevel='error', _apptag='UI')
    return metrics.get_metrics(), assess_info
