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
Tests for CAS hyperparameter tuning

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import re
import six
import swat
import swat.utils.testing as tm
import unittest
import pipefitter
from pipefitter.pipeline import Pipeline
from pipefitter.transformer import Imputer
from pipefitter.estimator import DecisionTree
from pipefitter.model_selection import HyperParameterTuning

from swat.utils.compat import patch_pandas_sort
from swat.utils.testing import UUID_RE, get_cas_host_type, load_data

patch_pandas_sort()

USER, PASSWD = tm.get_user_pass()
HOST, PORT, PROTOCOL = tm.get_host_port_proto()

# Classification
ctarget = 'Origin'

# Regression
rtarget = 'MSRP'

inputs = ['MPG_City', 'MPG_Highway', 'Length', 'Weight', 'Type', 'Cylinders']
nominals = ['Type', 'Cylinders', 'Origin']


class TestHyper(tm.TestCase):

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

    def test_basics(self, n_jobs=None):
        tbl = self.table

        estimator = DecisionTree(target='Origin', nominals=nominals,
                                 inputs=inputs)

        # dict of lists
        param_grid = dict(
            max_depth=[6, 10],
            leaf_size=[3, 5],
            alpha=0,
        )

        hpt = HyperParameterTuning(estimator=estimator,
                                   param_grid=param_grid)
        out = hpt.gridsearch(tbl, n_jobs=n_jobs)

        params = out['Parameters']

        res_params = sorted([{'max_depth': 6, 'leaf_size': 3, 'alpha': 0},
                             {'max_depth': 6, 'leaf_size': 5, 'alpha': 0},
                             {'max_depth': 10, 'leaf_size': 5, 'alpha': 0},
                             {'max_depth': 10, 'leaf_size': 3, 'alpha': 0}],
                            key=lambda x: (x['max_depth'], x['leaf_size']))

        self.assertEqual(sorted(list(params),
                                key=lambda x: (x['max_depth'], x['leaf_size'])),
                         res_params)

        self.assertEqual(len(out['FoldScores'][0]), 3)

        # list of dicts
        param_grid = [
            dict(max_depth=6, leaf_size=3, alpha=0),
            dict(max_depth=6, leaf_size=5, alpha=0),
            dict(max_depth=10, leaf_size=3, alpha=0),
            dict(max_depth=10, leaf_size=5, alpha=0),
        ]

        hpt = HyperParameterTuning(estimator=estimator,
                                   param_grid=param_grid)
        out = hpt.gridsearch(tbl, n_jobs=n_jobs)

        params = out['Parameters']

        res_params = sorted([{'max_depth': 6, 'leaf_size': 3, 'alpha': 0},
                             {'max_depth': 6, 'leaf_size': 5, 'alpha': 0},
                             {'max_depth': 10, 'leaf_size': 5, 'alpha': 0},
                             {'max_depth': 10, 'leaf_size': 3, 'alpha': 0}],
                            key=lambda x: (x['max_depth'], x['leaf_size']))
        
        self.assertEqual(sorted(list(params),
                                key=lambda x: (x['max_depth'], x['leaf_size'])),
                         res_params)

        self.assertEqual(len(out['FoldScores'][0]), 3)

    def test_interval(self):
        tbl = self.table

        estimator = DecisionTree(target='MSRP', nominals=nominals,
                                 inputs=inputs)

        # dict of lists
        param_grid = dict(
            max_depth=[6, 10],
            leaf_size=[3, 5],
            alpha=0,
        )

        hpt = HyperParameterTuning(estimator=estimator,
                                   param_grid=param_grid)
        out = hpt.gridsearch(tbl)

        params = out['Parameters']

        res_params = sorted([{'max_depth': 6, 'leaf_size': 3, 'alpha': 0},
                             {'max_depth': 6, 'leaf_size': 5, 'alpha': 0},
                             {'max_depth': 10, 'leaf_size': 5, 'alpha': 0},
                             {'max_depth': 10, 'leaf_size': 3, 'alpha': 0}],
                            key=lambda x: (x['max_depth'], x['leaf_size']))

        self.assertEqual(sorted(list(params),
                                key=lambda x: (x['max_depth'], x['leaf_size'])),
                         res_params)

        self.assertEqual(len(out['FoldScores'][0]), 3)

    def test_cv_iter(self):
        tbl = self.table

        estimator = DecisionTree(target='Origin', nominals=nominals,
                                 inputs=inputs)

        # dict of lists
        param_grid = dict(
            max_depth=[6, 10],
            leaf_size=[3, 5],
            alpha=0,
        )

        def cv_gen(tbl):
            yield tbl.sample(frac=0.1), tbl.sample(frac=0.9)
            yield tbl.sample(frac=0.2), tbl.sample(frac=0.8)
            yield tbl.sample(frac=0.3), tbl.sample(frac=0.7)
            yield tbl.sample(frac=0.4), tbl.sample(frac=0.6)

        test_cv = cv_gen(tbl)

        a, b = next(test_cv)
        self.assertEqual(len(a), 43)
        self.assertEqual(len(b), 385)

        a, b = next(test_cv)
        self.assertEqual(len(a), 86)
        self.assertEqual(len(b), 342)

        a, b = next(test_cv)
        self.assertEqual(len(a), 128)
        self.assertEqual(len(b), 300)

        a, b = next(test_cv)
        self.assertEqual(len(a), 171)
        self.assertEqual(len(b), 257)

        with self.assertRaises(StopIteration):
            a, b = next(test_cv)

        hpt = HyperParameterTuning(estimator=estimator,
                                   param_grid=param_grid,
                                   cv=cv_gen(tbl))
        out = hpt.gridsearch(tbl)

        params = out['Parameters']

        res_params = sorted([{'max_depth': 6, 'leaf_size': 3, 'alpha': 0},
                             {'max_depth': 6, 'leaf_size': 5, 'alpha': 0},
                             {'max_depth': 10, 'leaf_size': 5, 'alpha': 0},
                             {'max_depth': 10, 'leaf_size': 3, 'alpha': 0}],
                            key=lambda x: (x['max_depth'], x['leaf_size']))

        self.assertEqual(sorted(list(params),
                                key=lambda x: (x['max_depth'], x['leaf_size'])),
                         res_params)

        self.assertEqual(len(out['FoldScores'][0]), 4)

    def test_n_jobs(self):
         tbl = self.table

         # Session table can't do multiple jobs
         self.test_basics(n_jobs=3)

         # Promoted table can do multiple jobs
         try:
             tbl.table.promote()
             self.test_basics(n_jobs=3)
         finally:
             tbl.table.droptable()

    def test_pipeline(self):
        tbl = self.table

        modeimp = Imputer(Imputer.MODE)
        dtree1 = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)
        dtree2 = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)
        pipe = Pipeline([modeimp, dtree1, dtree2])

        # dict of lists
        param_grid = dict(
            max_depth=[6, 10],
            leaf_size=[3, 5],
            alpha=0,
        )

        hpt = HyperParameterTuning(estimator=pipe,
                                   param_grid=param_grid)
        out = hpt.gridsearch(tbl)

        params = out['Parameters']

        self.assertEqual(list(sorted(out.index)),
                         ['DecisionTree', 'DecisionTree',
                          'DecisionTree', 'DecisionTree',
                          'DecisionTree1', 'DecisionTree1',
                          'DecisionTree1', 'DecisionTree1'])

        res_params = sorted([{'max_depth': 6, 'leaf_size': 3, 'alpha': 0},
                             {'max_depth': 6, 'leaf_size': 3, 'alpha': 0},
                             {'max_depth': 6, 'leaf_size': 5, 'alpha': 0},
                             {'max_depth': 6, 'leaf_size': 5, 'alpha': 0},
                             {'max_depth': 10, 'leaf_size': 3, 'alpha': 0},
                             {'max_depth': 10, 'leaf_size': 3, 'alpha': 0},
                             {'max_depth': 10, 'leaf_size': 5, 'alpha': 0},
                             {'max_depth': 10, 'leaf_size': 5, 'alpha': 0}],
                            key=lambda x: (x['max_depth'], x['leaf_size']))

        self.assertEqual(sorted(list(params),
                                key=lambda x: (x['max_depth'], x['leaf_size'])),
                         res_params)

    def test_no_estimator_pipeline(self):
        tbl = self.table

        modeimp = Imputer(Imputer.MODE)
        pipe = Pipeline([modeimp])

        # dict of lists
        param_grid = dict(
            max_depth=[6, 10],
            leaf_size=[3, 5],
            alpha=0,
        )

        hpt = HyperParameterTuning(estimator=pipe,
                                   param_grid=param_grid)
        out = hpt.gridsearch(tbl)

        self.assertTrue(out is None)


if __name__ == '__main__':
    tm.runtests()
