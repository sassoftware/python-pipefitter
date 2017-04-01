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
Tests for CAS base classes

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import numpy as np
import pandas as pd
import swat
import swat.utils.testing as tm
import unittest
from pipefitter.backends.cas.estimator.base import EstimatorMixIn, ModelMixIn

from swat.utils.compat import patch_pandas_sort
from swat.utils.testing import UUID_RE, get_cas_host_type, load_data

patch_pandas_sort()

USER, PASSWD = tm.get_user_pass()
HOST, PORT, PROTOCOL = tm.get_host_port_proto()


class TestModelMixIn(tm.TestCase):

    def test_get_default_event_level(self):
        mmi = ModelMixIn()
        self.assertTrue(mmi.get_default_event_level(0) is None)
        self.assertTrue(mmi.get_default_event_level(2) is None)


class TestEstimatorMixIn(tm.TestCase):

    server_type = None

    def setUp(self):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        self.s = swat.CAS(HOST, PORT, USER, PASSWD, protocol=PROTOCOL)

        if type(self).server_type is None:
            type(self).server_type = get_cas_host_type(self.s)

        self.srcLib = tm.get_casout_lib(self.server_type)

        self.s.table.droptable('datasources.cars_single', caslib=self.srcLib,
                               _messagelevel='none')
        r = tm.load_data(self.s, 'datasources/cars_single.sashdat', self.server_type)

        self.table = r['casTable']

    def tearDown(self):
        # tear down tests
        self.s.table.droptable('datasources.cars_single', caslib=self.srcLib,
                               _messagelevel='none')
        self.s.terminate()
        del self.s
        swat.reset_option()

    def test_create_model_table(self):
        tbl = self.table

        est = EstimatorMixIn()

        out = est.create_model_table(self.s)
        self.assertEqual(out.__class__.__name__, 'CASTable')
        self.assertEqual(out.table.tableexists().exists, 0)

        # Test bad connection
        with self.assertRaises(ValueError):
            est.create_model_table(None)

        # Test caslib=
        out = est.create_model_table(self.s, caslib='casuser')
        self.assertEqual(out.caslib, 'casuser')

    def test_remap_params(self):
        est = EstimatorMixIn()
        out = est.remap_params({'foo':1, 'a.b':2, 'c':100}, {'c':3})
        self.assertEqual(out, dict(foo=1, a=dict(b=2), c=3))
        
        out = est.remap_params({'foo':1, 'a.b':2, 'c':100}, {'c':3, 'a.d':4})
        self.assertEqual(out, dict(foo=1, a=dict(b=2, d=4), c=3))
        

if __name__ == '__main__':
    tm.runtests()
