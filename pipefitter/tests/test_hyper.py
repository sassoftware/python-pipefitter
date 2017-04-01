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
Tests for hyperparameter tuning

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import re
import six
import swat.utils.testing as tm
import unittest
import pipefitter
from pipefitter.estimator import DecisionTree
from pipefitter.model_selection import HyperParameterTuning


class TestHyper(tm.TestCase):

    def test_params(self):
        estimator = DecisionTree()
        param_grid=dict(
            max_depth=[6, 10],
            leaf_size=[3, 5],
        )

        # Basic settings and defaults
        hpt = HyperParameterTuning(estimator=estimator,
                                   param_grid=param_grid)
        self.assertEqual(hpt.params['estimator'], estimator)
        self.assertEqual(hpt.params['param_grid'], param_grid)
        self.assertEqual(hpt.params['cv'], 3)
        self.assertTrue(hpt.params['score_type'] is None)

        # cv = int
        hpt = HyperParameterTuning(estimator=estimator,
                                   param_grid=param_grid,
                                   cv=3)
        self.assertEqual(hpt.params['cv'], 3)

        # cv = float
        hpt = HyperParameterTuning(estimator=estimator,
                                   param_grid=param_grid,
                                   cv=0.3)
        self.assertEqual(hpt.params['cv'], 0.3)

        # cv = -float
        with self.assertRaises(ValueError):
            hpt = HyperParameterTuning(estimator=estimator,
                                       param_grid=param_grid, cv=-0.1)

        # cv = float > 1
        with self.assertRaises(ValueError):
            hpt = HyperParameterTuning(estimator=estimator,
                                       param_grid=param_grid, cv=1.0001)

        # cv = generator
        gen = iter([0])
        hpt = HyperParameterTuning(estimator=estimator,
                                   param_grid=param_grid,
                                   cv=gen)
        self.assertEqual(hpt.params['cv'], gen)

        # cv = list
        items = [0]
        hpt = HyperParameterTuning(estimator=estimator,
                                   param_grid=param_grid,
                                   cv=items)
        self.assertEqual(hpt.params['cv'], items)

        # cv = string
        with self.assertRaises(TypeError):
            HyperParameterTuning(estimator=estimator,
                                 param_grid=param_grid,
                                 cv='foo')

        # cv = 1 (lower than minimum)
        with self.assertRaises(ValueError):
            hpt = HyperParameterTuning(estimator=estimator,
                                       param_grid=param_grid, cv=1)


if __name__ == '__main__':
    tm.runtests()
