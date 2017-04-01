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
Tests for CAS Pipeline

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import numpy as np
import pandas as pd
import swat
import swat.utils.testing as tm
import unittest
from pipefitter.estimator import DecisionTree, DecisionForest, GBTree
from pipefitter.pipeline import Pipeline, tosequence
from pipefitter.transformer import Imputer

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


class TestPipelineUtils(tm.TestCase):

    def test_tosequence(self):
        self.assertEqual(tosequence(('a', 'b', 'c')), ('a', 'b', 'c'))
        self.assertEqual(tosequence(['a', 'b', 'c']), ['a', 'b', 'c'])
        self.assertEqual(tosequence(iter(('a', 'b', 'c'))), ['a', 'b', 'c'])
        self.assertEqual(tosequence('abc'), 'abc')
        self.assertEqual(list(tosequence(np.array((1, 2, 3)))),
                         list(np.asarray(np.array((1, 2, 3)))))

        with self.assertRaises(TypeError):
            tosequence(4)

class TestPipeline(tm.TestCase):

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

    def test_basic(self):
        tbl = self.table

        mean_imp = Imputer(Imputer.MEAN)
        mode_imp = Imputer(Imputer.MODE)
        dtree = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)

        pipe = Pipeline([mean_imp, mode_imp, dtree])

        model = pipe.fit(tbl)
        self.assertEqual(model.__class__.__name__, 'PipelineModel')
        self.assertEqual(len(model.stages), 3)
        self.assertTrue(model[0] is mean_imp)
        self.assertTrue(model[1] is mode_imp)
        self.assertEqual(model[2].__class__.__name__, 'DecisionTreeModel')

        out = model.score(tbl)

        self.assertEqual(set(list(out.index)), 
                         set(['Target', 'Level', 'Var', 'NBins', 'NObsUsed',
                              'TargetCount', 'TargetMiss', 'PredCount', 'PredMiss',
                              'Event', 'EventCount', 'NonEventCount', 'EventMiss',
                              'AreaUnderROCCurve', 'CRCut', 'ClassificationCutOff',
                              'KS', 'KSCutOff', 'MisClassificationRate']))

        # Bad item type
        with self.assertRaises(TypeError): 
            Pipeline([mean_imp, mode_imp, 'foo', dtree])

    def test_multiple_estimators(self):
        tbl = self.table

        mean_imp = Imputer(Imputer.MEAN)
        mode_imp = Imputer(Imputer.MODE)
        dtree1 = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)
        dtree2 = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)

        pipe = Pipeline([mean_imp, mode_imp, dtree1, dtree2])

        model = pipe.fit(tbl)
        self.assertEqual(model.__class__.__name__, 'PipelineModel')
        self.assertEqual(len(model.stages), 4)
        self.assertTrue(model[0] is mean_imp)
        self.assertTrue(model[1] is mode_imp)
        self.assertEqual(model[2].__class__.__name__, 'DecisionTreeModel')
        self.assertEqual(model[3].__class__.__name__, 'DecisionTreeModel')

        out = model.score(tbl)
        self.assertEqual(set(list(out.index)),
                         set(['DecisionTree', 'DecisionTree1']))

    def test_str(self):
        mean_imp = Imputer(Imputer.MEAN)
        mode_imp = Imputer(Imputer.MODE)
        dtree = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)
        pipe = Pipeline([mean_imp, mode_imp, dtree])
        
        out = "Pipeline([Imputer(MEAN), Imputer(MODE), " + \
              "DecisionTree(alpha=0.0, cf_level=0.25, criterion=None, " + \
              "inputs=['MPG_City', 'MPG_Highway', 'Length', 'Weight', " + \
              "'Type', 'Cylinders'], leaf_size=5, max_branches=2, " + \
              "max_depth=6, n_bins=20, nominals=['Type', 'Cylinders', " + \
              "'Origin'], prune=False, target='Origin', var_importance=False)])"

        self.assertEqual(str(pipe).replace("u'", "'"), out)

    def test_repr(self):
        mean_imp = Imputer(Imputer.MEAN)
        mode_imp = Imputer(Imputer.MODE)
        dtree = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)
        pipe = Pipeline([mean_imp, mode_imp, dtree])

        out = "Pipeline([Imputer(MEAN), Imputer(MODE), " + \
              "DecisionTree(alpha=0.0, cf_level=0.25, criterion=None, " + \
              "inputs=['MPG_City', 'MPG_Highway', 'Length', 'Weight', " + \
              "'Type', 'Cylinders'], leaf_size=5, max_branches=2, " + \
              "max_depth=6, n_bins=20, nominals=['Type', 'Cylinders', " + \
              "'Origin'], prune=False, target='Origin', var_importance=False)])"

        self.assertEqual(repr(pipe).replace("u'", "'"), out)

    def test_model_str(self):
        tbl = self.table

        mean_imp = Imputer(Imputer.MEAN)
        mode_imp = Imputer(Imputer.MODE)
        dtree = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)
        model = Pipeline([mean_imp, mode_imp, dtree]).fit(tbl)

        out = "PipelineModel([Imputer(MEAN), Imputer(MODE), " + \
              "DecisionTreeModel(alpha=0.0, cf_level=0.25, criterion=None, " + \
              "inputs=['MPG_City', 'MPG_Highway', 'Length', 'Weight', " + \
              "'Type', 'Cylinders'], leaf_size=5, max_branches=2, " + \
              "max_depth=6, n_bins=20, nominals=['Type', 'Cylinders', " + \
              "'Origin'], prune=False, target='Origin', var_importance=False)])"

        self.assertEqual(str(model).replace("u'", "'"), out)

    def test_model_repr(self):
        tbl = self.table

        mean_imp = Imputer(Imputer.MEAN)
        mode_imp = Imputer(Imputer.MODE)
        dtree = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)
        model = Pipeline([mean_imp, mode_imp, dtree]).fit(tbl)

        out = "PipelineModel([Imputer(MEAN), Imputer(MODE), " + \
              "DecisionTreeModel(alpha=0.0, cf_level=0.25, criterion=None, " + \
              "inputs=['MPG_City', 'MPG_Highway', 'Length', 'Weight', " + \
              "'Type', 'Cylinders'], leaf_size=5, max_branches=2, " + \
              "max_depth=6, n_bins=20, nominals=['Type', 'Cylinders', " + \
              "'Origin'], prune=False, target='Origin', var_importance=False)])"

        self.assertEqual(repr(model).replace("u'", "'"), out)

    def test_set_params(self):
        tbl = self.table

        mean_imp = Imputer(Imputer.MEAN)
        mode_imp = Imputer(Imputer.MODE)
        dtree = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)

        pipe = Pipeline([mean_imp, mode_imp, dtree])
        out = pipe.fit(tbl).score(tbl)
        self.assertEqual(out.loc['Target'], 'Origin')

        # Set extra parameters on Pipeline (not on estimator)
        pipe.set_params({dtree.target: 'MSRP'})
        self.assertEqual(dtree.target, 'Origin')

        out = pipe.fit(tbl).score(tbl)
        self.assertEqual(out.loc['Target'], 'MSRP')

        # Set parameters during fit
        pipe = Pipeline([mean_imp, mode_imp, dtree])

        out = pipe.fit(tbl).score(tbl)
        self.assertEqual(out.loc['Target'], 'Origin')

        out = pipe.fit(tbl, {dtree.target: 'MSRP'}).score(tbl)
        self.assertEqual(out.loc['Target'], 'MSRP')

    def test_transform(self):
        tbl = self.table

        mode_imp = Imputer(Imputer.MODE)
        dtree = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)

        pipe = Pipeline([mode_imp, dtree])

        self.assertEqual(tbl.nmiss().max(), 2)

        out = pipe.transform(tbl)

        self.assertEqual(out.__class__.__name__, 'CASTable') 
        self.assertEqual(tbl.nmiss().max(), 2)
        self.assertEqual(out.nmiss().max(), 0)

    def test_model_transform(self):
        tbl = self.table

        mode_imp = Imputer(Imputer.MODE)
        dtree = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)

        pipe = Pipeline([mode_imp, dtree])

        self.assertEqual(tbl.nmiss().max(), 2)

        model = pipe.fit(tbl)
        out = model.transform(tbl)

        self.assertEqual(out.__class__.__name__, 'CASTable')
        self.assertEqual(tbl.nmiss().max(), 2)
        self.assertEqual(out.nmiss().max(), 0)

    def test_getitem(self):
        tbl = self.table

        mode_imp = Imputer(Imputer.MODE)
        dtree = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)
        
        pipe = Pipeline([mode_imp, dtree])

        self.assertTrue(pipe[0] is mode_imp)
        self.assertTrue(pipe[1] is dtree)

        with self.assertRaises(IndexError):
            pipe[2]

        with self.assertRaises(TypeError):
            pipe['foo']

    def test_model_getitem(self):
        tbl = self.table

        mode_imp = Imputer(Imputer.MODE)
        dtree = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)

        model = Pipeline([mode_imp, dtree]).fit(tbl)

        self.assertTrue(model[0] is mode_imp)
        self.assertTrue(model[1] is not dtree)
        self.assertEqual(model[1].__class__.__name__, 'DecisionTreeModel')

        with self.assertRaises(IndexError):
            model[2]

        with self.assertRaises(TypeError):
            model['foo']

    def test_classification_score(self):
        tbl = self.table

        mean_imp = Imputer(Imputer.MEAN)
        mode_imp = Imputer(Imputer.MODE)
        dtree = DecisionTree(target='Origin', nominals=nominals, inputs=inputs)

        pipe = Pipeline([mean_imp, mode_imp, dtree])

        model = pipe.fit(tbl)
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

        mean_imp = Imputer(Imputer.MEAN)
        mode_imp = Imputer(Imputer.MODE)
        dtree = DecisionTree(target='MSRP', nominals=nominals, inputs=inputs)

        pipe = Pipeline([mean_imp, mode_imp, dtree])

        model = pipe.fit(tbl)
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
        mean_imp = Imputer(Imputer.MEAN)
        mode_imp = Imputer(Imputer.MODE)
        dtree = DecisionTree(target='MSRP', nominals=nominals, inputs=inputs)

        pipe = Pipeline([mean_imp, mode_imp, dtree])

        model = pipe.fit(self.table)
        self.assertEqual(model[-1].data.table.tableexists().exists, 1)
        model.unload()
        self.assertEqual(model[-1].data.table.tableexists().exists, 0)


if __name__ == '__main__':
    tm.runtests()
