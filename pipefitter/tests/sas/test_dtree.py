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
Tests for CAS Estimators

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import numpy as np
import pandas as pd
import saspy
import unittest
from pipefitter.estimator import DecisionTree


class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        # Use the first entry in the configuration list
        self.sas = saspy.SASsession()
        self.assertIsInstance(self.sas, saspy.SASsession, msg="sas = saspy.SASsession(...) failed")
        self.table = self.sas.sasdata('cars', 'sashelp')

    def tearDown(self):
        if self.sas:
           self.sas._endsas()

    @unittest.skip('Not implemented')
    def test_params(self):
        tbl = self.table

        # Check defaults
        dtree = DecisionTree()
        self.assertEqual(dtree.params.to_dict(), dtree_defaults)

        # Check constructor parameters
        params = dtree_defaults.copy()
        params.update(dict(prune=True, target='Cylinders', nominals=['Make', 'Model'],
                           inputs=['Make', 'Model', 'Horsepower']))
        dtree = DecisionTree(prune=True, target='Cylinders', nominals=['Make', 'Model'],
                             inputs=['Make', 'Model', 'Horsepower'])
        self.assertEqual(dtree.params.to_dict(), params)

        model = dtree.fit(tbl)
        self.assertEqual(model.__class__.__name__, 'DecisionTreeModel')
        self.assertEqual(model.params, params)

        # Check constructor parameter error
        with self.assertRaises(ValueError):
            DecisionTree(prune=True, criterion='foo',
                         target='Cylinders', nominals=['Make', 'Model'],
                         inputs=['Make', 'Model', 'Horsepower'])

        with self.assertRaises(TypeError):
            DecisionTree(foo='bar')

        # Check fit parameter overrides
        params = dtree_defaults.copy()
        params.update(dict(max_depth=7, leaf_size=5,
                           target='Cylinders', nominals=['Make', 'Model'],
                           inputs=['Make', 'Model', 'Horsepower']))

        model = dtree.fit(tbl, prune=False, max_depth=7)
        self.assertEqual(model.__class__.__name__, 'DecisionTreeModel')
        self.assertEqual(model.params, params)

        # Check parameter overrides error
        with self.assertRaises(TypeError):
            dtree.fit(tbl, prune='foo', max_depth=7)

        with self.assertRaises(KeyError):
            dtree.fit(tbl, foo='bar')

    @unittest.skip('Not implemented')
    def test_fit(self):
        tbl = self.table

        params = dtree_defaults.copy()
        params.update(dict(target='Cylinders', nominals=['Make', 'Model'],
                           inputs=['Make', 'Model', 'Horsepower']))

        dtree = DecisionTree(target='Cylinders', nominals=['Make', 'Model'],
                             inputs=['Make', 'Model', 'Horsepower'])
        model = dtree.fit(tbl)

        self.assertEqual(model.__class__.__name__, 'DecisionTreeModel')
        self.assertEqual(model.data.__class__.__name__, 'CASTable')
        self.assertEqual(model.params, params)
        self.assertEqual(model.diagnostics.__class__.__name__, 'CASResults')
        self.assertEqual(sorted(model.diagnostics.keys()), ['ModelInfo', 'OutputCasTables'])

        # Have nominals set automatically
        dtree = DecisionTree(target='Cylinders', nominals=[],
                             inputs=['Make', 'Model', 'Horsepower'])
        model = dtree.fit(tbl)
        self.assertEqual(model.params['nominals'], [])

    @unittest.skip('Not implemented')
    def test_score(self):
        tbl = self.table

        params = dtree_defaults.copy()
        params.update(dict(target='Cylinders', nominals=['Make', 'Model'],
                           inputs=['Make', 'Model', 'Horsepower']))

        dtree = DecisionTree(target='Cylinders', nominals=['Make', 'Model'],
                             inputs=['Make', 'Model', 'Horsepower'])
        model = dtree.fit(tbl)
        score = model.score(tbl)
        self.assertTrue(isinstance(score, pd.Series))
        self.assertAlmostEqual(score.loc['MeanSquaredError'], 0.4423817642)
        self.assertEqual(score.loc['NObsUsed'], 426)
        self.assertEqual(score.loc['NObsRead'], 428)

    @unittest.skip('Not implemented')
    def test_unload(self):
        dtree = DecisionTree(target='Cylinders', nominals=['Make', 'Model'],
                             inputs=['Make', 'Model', 'Horsepower'])
        model = dtree.fit(self.table)
        self.assertEqual(model.data.table.tableexists().exists, 1)
        model.unload()
        self.assertEqual(model.data.table.tableexists().exists, 0)

if __name__ == '__main__':
    unittest.main()
