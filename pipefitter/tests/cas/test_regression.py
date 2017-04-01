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
Tests for CAS Regression Estimators

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import numpy as np
import pandas as pd
import swat
import swat.utils.testing as tm
import unittest
from pipefitter.estimator import LogisticRegression, LinearRegression

from swat.utils.compat import patch_pandas_sort
from swat.utils.testing import UUID_RE, get_cas_host_type, load_data

patch_pandas_sort()

USER, PASSWD = tm.get_user_pass()
HOST, PORT, PROTOCOL = tm.get_host_port_proto()

logistic_defaults = dict(intercept=True, max_effects=0, selection='none', sig_level=0.05, 
                         criterion=None, target=None, nominals=[], inputs=[])

linear_defaults = dict(intercept=True, max_effects=0, selection='none', sig_level=0.05, 
                       criterion=None, target=None, nominals=[], inputs=[])

# Classification
ctarget = 'Origin'

# Regression
rtarget = 'MSRP'

inputs = ['MPG_City', 'MPG_Highway', 'Length', 'Weight', 'Type', 'Cylinders']
nominals = ['Type']


class TestLogistic(tm.TestCase):

    server_type = None

    def setUp(self):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        self.s = swat.CAS(HOST, PORT, USER, PASSWD, protocol=PROTOCOL)

        if type(self).server_type is None:
            type(self).server_type = get_cas_host_type(self.s)

        self.srcLib = tm.get_casout_lib(self.server_type)

        r = tm.load_data(self.s, 'datasources/cars_single.sashdat', self.server_type)

        self.table = r['casTable']

    def tearDown(self):
        # tear down tests
        self.s.terminate()
        del self.s
        swat.reset_option()

    def test_params(self):
        tbl = self.table

        # Check defaults
        log = LogisticRegression()
        self.assertEqual(log.params.to_dict(), logistic_defaults)

        # Check constructor parameters
        params = logistic_defaults.copy()
        params.update(dict(intercept=True, selection='backward',
                           target='Origin', nominals=nominals, inputs=inputs))
        log = LogisticRegression(intercept=True, selection='backward',
                                 target='Origin', nominals=nominals, inputs=inputs)
        self.assertEqual(log.params.to_dict(), params)

        model = log.fit(tbl)
        self.assertEqual(model.__class__.__name__, 'LogisticRegressionModel')
        self.assertEqual(model.params, params)
       
        # Check constructor parameter error
        with self.assertRaises(ValueError):
            LogisticRegression(intercept=10, target='Origin',
                               nominals=nominals, inputs=inputs)

        with self.assertRaises(TypeError):
            LogisticRegression(foo='bar')

        # Check fit parameter overrides
        params = logistic_defaults.copy()
        params.update(dict(intercept=False, criterion='aic', selection='backward',
                           target='Origin', nominals=nominals, inputs=inputs))

        model = log.fit(tbl, intercept=False, criterion='aic')
        self.assertEqual(model.__class__.__name__, 'LogisticRegressionModel')
        self.assertEqual(model.params, params)

        # Check parameter overrides error
        with self.assertRaises(TypeError):
            log.fit(tbl, intercept='foo', criterion='aic')

        with self.assertRaises(KeyError):
            log.fit(tbl, foo='bar') 

    def test_selection(self):
        # This test merely makes sure that the `fit` method will run with
        # all of the specified selection methods.
        tbl = self.table

        params = logistic_defaults.copy()
        params.update(dict(selection='none', max_effects=0,
                           criterion=None, intercept=True,
                           target='Origin', nominals=nominals, inputs=inputs))

        # criterion='sl' not compatible with selection='backward' | 'lasso'
        with self.assertRaises(ValueError):
            LogisticRegression(selection='backward', criterion='sl',
                               target='Origin', nominals=nominals, inputs=inputs)

        with self.assertRaises(ValueError):
            LogisticRegression(selection='lasso', criterion='sl',
                               target='Origin', nominals=nominals, inputs=inputs)

    def test_fit(self):
        tbl = self.table

        params = logistic_defaults.copy()
        params.update(dict(target='Origin', nominals=nominals, inputs=inputs))

        log = LogisticRegression(target='Origin', nominals=nominals, inputs=inputs)
        model = log.fit(tbl)

        self.assertEqual(model.__class__.__name__, 'LogisticRegressionModel')
        self.assertTrue(model.data, dict)
        self.assertTrue('dscode' in model.data)
        self.assertEqual(model.params, params)
        self.assertEqual(model.diagnostics.__class__.__name__, 'CASResults')
        self.assertEqual(sorted(model.diagnostics.keys()), ['ClassInfo',
             'ConvergenceStatus', 'Dimensions', 'FitStatistics', 'GlobalTest',
             'ModelInfo', 'NObs', 'ParameterEstimates', 'ResponseProfile', 'Timing'])

    def test_classification_score(self):
        tbl = self.table

        params = logistic_defaults.copy()
        params.update(dict(target='Origin', nominals=nominals, inputs=inputs))

        log = LogisticRegression(target='Origin', nominals=nominals, inputs=inputs)
        model = log.fit(tbl)
        score = model.score(tbl)
        self.assertTrue(isinstance(score, pd.Series))
        self.assertEqual(score.loc['NObsUsed'], 428)
        #self.assertEqual(score.loc['NObsRead'], 428)

    @unittest.skip('The logistic action hangs')
    def test_regression_score(self):
        tbl = self.table

        params = logistic_defaults.copy()
        params.update(dict(target='MSRP', nominals=nominals, inputs=inputs))

        log = LogisticRegression(target='MSRP', nominals=nominals, inputs=inputs)
        model = log.fit(tbl)
        score = model.score(tbl)
        self.assertTrue(isinstance(score, pd.Series))
        self.assertEqual(score.loc['NObsUsed'], 428)
        #self.assertEqual(score.loc['NObsRead'], 428)

    def test_unload(self):
        log = LogisticRegression(target='Origin', nominals=nominals, inputs=inputs)
        model = log.fit(self.table)
        model.unload()


class TestLinear(tm.TestCase):

    server_type = None

    def setUp(self):
        swat.reset_option()

        self.s = swat.CAS(HOST, PORT, USER, PASSWD, protocol=PROTOCOL)

        if type(self).server_type is None:
            type(self).server_type = get_cas_host_type(self.s)

        self.srcLib = tm.get_casout_lib(self.server_type)

        r = tm.load_data(self.s, 'datasources/cars_single.sashdat', self.server_type)

        self.table = r['casTable']

    def tearDown(self):
        # tear down tests
        self.s.terminate()
        del self.s
        swat.reset_option()

    def test_params(self):
        tbl = self.table

        # Check defaults
        lin = LinearRegression()
        self.assertEqual(lin.params.to_dict(), linear_defaults)

        # Check constructor parameters
        params = linear_defaults.copy()
        params.update(dict(intercept=True, selection='backward',
                           target='MSRP', nominals=nominals, inputs=inputs))
        lin = LinearRegression(intercept=True, selection='backward',
                               target='MSRP', nominals=nominals, inputs=inputs)
        self.assertEqual(lin.params.to_dict(), params)

        model = lin.fit(tbl)
        self.assertEqual(model.__class__.__name__, 'LinearRegressionModel')
        self.assertEqual(model.params, params)
       
        # Check constructor parameter error
        with self.assertRaises(ValueError):
            LinearRegression(intercept=10, target='MSRP',
                             nominals=nominals, inputs=inputs)

        with self.assertRaises(TypeError):
            LinearRegression(foo='bar')

        # Check fit parameter overrides
        params = linear_defaults.copy()
        params.update(dict(intercept=False, criterion='aic', selection='backward',
                           target='MSRP', nominals=nominals, inputs=inputs))

        model = lin.fit(tbl, intercept=False, criterion='aic')
        self.assertEqual(model.__class__.__name__, 'LinearRegressionModel')
        self.assertEqual(model.params, params)

        # Check parameter overrides error
        with self.assertRaises(TypeError):
            lin.fit(tbl, intercept='foo', criterion='aic')

        with self.assertRaises(KeyError):
            lin.fit(tbl, foo='bar') 

    def test_fit(self):
        tbl = self.table

        params = linear_defaults.copy()
        params.update(dict(target='MSRP', nominals=nominals, inputs=inputs))

        lin = LinearRegression(target='MSRP', nominals=nominals, inputs=inputs)
        model = lin.fit(tbl)

        self.assertEqual(model.__class__.__name__, 'LinearRegressionModel')
        self.assertTrue(model.data, dict)
        self.assertTrue('dscode' in model.data)
        self.assertEqual(model.params, params)
        self.assertEqual(model.diagnostics.__class__.__name__, 'CASResults')
        self.assertEqual(sorted(model.diagnostics.keys()), 
                         ['ANOVA', 'ClassInfo', 'Dimensions', 'FitStatistics',
                          'ModelInfo', 'NObs', 'ParameterEstimates', 'Timing'])

    def test_selection(self):
        # This test merely makes sure that the `fit` method will run with
        # all of the specified selection methods.
        tbl = self.table

        params = linear_defaults.copy()
        params.update(dict(selection='none', max_effects=0, 
                           criterion=None, intercept=True,
                           target='MSRP', nominals=nominals, inputs=inputs))

        # none (default)
        lin = LinearRegression(target='MSRP', nominals=nominals, inputs=inputs)
        model = lin.fit(tbl)
        self.assertEqual(model.params, params)

        # none 
        lin = LinearRegression(selection='none', target='MSRP',
                               nominals=nominals, inputs=inputs)
        model = lin.fit(tbl)
        self.assertEqual(model.params, params)

        # forward
        lin = LinearRegression(selection='forward', target='MSRP',
                               nominals=nominals, inputs=inputs)
        params.update(dict(selection='forward'))
        model = lin.fit(tbl)
        self.assertEqual(model.params, params)

        # backward
        lin = LinearRegression(selection='backward', target='MSRP',
                               nominals=nominals, inputs=inputs)
        model = lin.fit(tbl)
        params.update(dict(selection='backward'))
        self.assertEqual(model.params, params)

        # stepwise
        lin = LinearRegression(selection='stepwise', target='MSRP',
                               nominals=nominals, inputs=inputs)
        model = lin.fit(tbl)
        params.update(dict(selection='stepwise'))
        self.assertEqual(model.params, params)

        # lasso
        lin = LinearRegression(selection='lasso', target='MSRP',
                               nominals=nominals, inputs=inputs)
        model = lin.fit(tbl)
        params.update(dict(selection='lasso'))
        self.assertEqual(model.params, params)

        #
        # Check selection methods with criterion='sl'
        #

        params.update(dict(selection='none', criterion='sl'))

        # none (default)
        lin = LinearRegression(criterion='sl', target='MSRP',
                               nominals=nominals, inputs=inputs)
        model = lin.fit(tbl)
        self.assertEqual(model.params, params)

        # none
        lin = LinearRegression(selection='none', criterion='sl',
                               target='MSRP', nominals=nominals,
                               inputs=inputs)
        model = lin.fit(tbl)
        self.assertEqual(model.params, params)

        # forward
        lin = LinearRegression(selection='forward', criterion='sl',
                               target='MSRP', nominals=nominals,
                               inputs=inputs)
        params.update(dict(selection='forward'))
        model = lin.fit(tbl)
        self.assertEqual(model.params, params)

        # backward
        with self.assertRaises(ValueError):
            lin = LinearRegression(selection='backward', criterion='sl',
                                   target='MSRP', nominals=nominals,
                                   inputs=inputs)

        # stepwise
        lin = LinearRegression(selection='stepwise', criterion='sl',
                               target='MSRP', nominals=nominals,
                               inputs=inputs)
        model = lin.fit(tbl)
        params.update(dict(selection='stepwise'))
        self.assertEqual(model.params, params)

        # lasso
        with self.assertRaises(ValueError):
            lin = LinearRegression(selection='lasso', criterion='sl',
                                   target='MSRP', nominals=nominals,
                                   inputs=inputs)

    def test_regression_score(self):
        tbl = self.table

        params = linear_defaults.copy()
        params.update(dict(target='MSRP', nominals=nominals, inputs=inputs))

        lin = LinearRegression(target='MSRP', nominals=nominals, inputs=inputs)
        model = lin.fit(tbl)
        score = model.score(tbl)
        self.assertTrue(isinstance(score, pd.Series))
        self.assertEqual(score.loc['Target'], 'MSRP')
        self.assertEqual(score.loc['Level'], 'INTERVAL')
        self.assertEqual(score.loc['NBins'], 100)
        self.assertEqual(score.loc['NObsUsed'], 428)
        self.assertTrue(isinstance(score.loc['AverageSquaredError'], float))
        self.assertTrue(isinstance(score.loc['AverageAbsoluteError'], float))
        self.assertTrue(isinstance(score.loc['AverageSquaredLogarithmicError'], float))
        self.assertTrue(isinstance(score.loc['RootAverageSquaredError'], float))
        self.assertTrue(isinstance(score.loc['RootAverageAbsoluteError'], float))
        self.assertTrue(isinstance(score.loc['RootAverageSquaredLogarithmicError'], float))

    def test_classification_score(self):
        tbl = self.table

        params = linear_defaults.copy()
        params.update(dict(target='Origin', nominals=nominals, inputs=inputs))

        lin = LinearRegression(target='Origin', nominals=nominals, inputs=inputs)
        with self.assertRaises(RuntimeError):
            lin.fit(tbl)

    def test_unload(self):
        lin = LinearRegression(target='MSRP', nominals=nominals, inputs=inputs)
        model = lin.fit(self.table)
        model.unload()


if __name__ == '__main__':
    tm.runtests()
