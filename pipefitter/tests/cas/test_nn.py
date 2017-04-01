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
Tests for Neural Networks

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import numpy as np
import pandas as pd
import swat
import swat.utils.testing as tm
import unittest
from pipefitter.estimator import NeuralNetwork

from swat.utils.compat import patch_pandas_sort
from swat.utils.testing import UUID_RE, get_cas_host_type, load_data

patch_pandas_sort()

USER, PASSWD = tm.get_user_pass()
HOST, PORT, PROTOCOL = tm.get_host_port_proto()

nn_defaults = {'annealing_rate': 1e-06, 'hiddens': [9],
               'direct': False, 'inputs': [], 'error_func': None,
               'lasso': 0.0, 'std': 'midrange',
               'max_iters': 10, 'acts': 'tanh', 'max_time': 0.0,
               'nominals': [], 'ridge': 0.0, 'seed': 0.0,
               'learning_rate': 0.001, 'target': None,
               'num_tries': 10, 'optimization': 'lbfgs'}

# Classification
ctarget = 'Origin'

# Regression
rtarget = 'MSRP'

inputs = ['MPG_City', 'MPG_Highway', 'Length', 'Weight', 'Type', 'Cylinders']
nominals = ['Type', 'Cylinders', 'Origin']


class TestNeuralNetwork(tm.TestCase):

    server_type = None

    def setUp(self):
        swat.reset_option()
        swat.options.cas.print_messages = True
        swat.options.interactive_mode = True

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
        nn = NeuralNetwork()
        self.assertEqual(nn.params.to_dict(), nn_defaults)

        # Check constructor parameters
        params = nn_defaults.copy()
        params.update(dict(direct=True, target='Origin', nominals=nominals, inputs=inputs))
        nn = NeuralNetwork(direct=True, target='Origin', nominals=nominals, inputs=inputs)
        self.assertEqual(nn.params.to_dict(), params)

        model = nn.fit(tbl)
        self.assertEqual(model.__class__.__name__, 'NeuralNetworkModel')
        self.assertEqual(model.params, params)
       
        # Check constructor parameter error
        with self.assertRaises(ValueError):
            NeuralNetwork(direct=True, std='foo',
                           target='Origin', nominals=nominals, inputs=inputs)

        with self.assertRaises(TypeError):
            NeuralNetwork(foo='bar')

        # Check fit parameter overrides
        params = nn_defaults.copy()
        params.update(dict(direct=False, max_iters=10,
                           target='Origin', nominals=nominals, inputs=inputs))

        model = nn.fit(tbl, direct=False, max_iters=10)
        self.assertEqual(model.__class__.__name__, 'NeuralNetworkModel')
        self.assertEqual(model.params, params)

        # Check parameter overrides error
        with self.assertRaises(TypeError):
            nn.fit(tbl, direct='foo', max_iters=7)

        with self.assertRaises(KeyError):
            nn.fit(tbl, foo='bar') 

    def test_fit(self):
        tbl = self.table

        params = nn_defaults.copy()
        params.update(dict(target='Origin', nominals=nominals, inputs=inputs))

        nn = NeuralNetwork(target='Origin', nominals=nominals, inputs=inputs)
        model = nn.fit(tbl)

        self.assertEqual(model.__class__.__name__, 'NeuralNetworkModel')
        self.assertEqual(model.data.__class__.__name__, 'CASTable')
        self.assertEqual(model.params, params)
        self.assertEqual(model.diagnostics.__class__.__name__, 'CASResults')
        self.assertEqual(sorted(model.diagnostics.keys()), 
                         ['ConvergenceStatus0', 'ConvergenceStatus1',
                          'ConvergenceStatus2', 'ConvergenceStatus3',
                          'ConvergenceStatus4', 'ConvergenceStatus5',
                          'ConvergenceStatus6', 'ConvergenceStatus7',
                          'ConvergenceStatus8', 'ConvergenceStatus9',
                          'ModelInfo',
                          'OptIterHistory0', 'OptIterHistory1',
                          'OptIterHistory2', 'OptIterHistory3',
                          'OptIterHistory4', 'OptIterHistory5',
                          'OptIterHistory6', 'OptIterHistory7',
                          'OptIterHistory8', 'OptIterHistory9',
                          'OutputCasTables'])

    def test_classification_score(self):
        tbl = self.table

        params = nn_defaults.copy()
        params.update(dict(target='Origin', nominals=nominals, inputs=inputs))

        nn = NeuralNetwork(target='Origin', nominals=nominals, inputs=inputs)
        model = nn.fit(tbl)
        score = model.score(tbl)
        self.assertTrue(isinstance(score, pd.Series))
        self.assertEqual(score.loc['Target'], 'Origin')
        self.assertEqual(score.loc['Level'], 'CLASS')
        self.assertEqual(score.loc['Event'], 'USA')
        self.assertEqual(score.loc['NBins'], 100)
        self.assertEqual(score.loc['NObsUsed'], 428)
        self.assertTrue(isinstance(score.loc['AreaUnderROCCurve'], float))
        self.assertTrue(isinstance(score.loc['CRCut'], float))
        self.assertTrue(isinstance(score.loc['KS'], float))
        self.assertTrue(isinstance(score.loc['KSCutOff'], float))
        self.assertTrue(isinstance(score.loc['MisClassificationRate'], float))

    def test_regression_score(self):
        tbl = self.table

        params = nn_defaults.copy()
        params.update(dict(target='MSRP', nominals=nominals, inputs=inputs))

        nn = NeuralNetwork(target='MSRP', nominals=nominals, inputs=inputs)
        model = nn.fit(tbl)
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

    def test_unload(self):
        nn = NeuralNetwork(target='Origin', nominals=nominals, inputs=inputs)
        model = nn.fit(self.table)
        self.assertEqual(model.data.table.tableexists().exists, 1)
        model.unload()
        self.assertEqual(model.data.table.tableexists().exists, 0)


if __name__ == '__main__':
    tm.runtests()
