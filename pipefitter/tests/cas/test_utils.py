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
import swat
import swat.utils.testing as tm
import unittest
from pipefitter.backends.cas.utils import ResourceManager
from pipefitter.estimator import DecisionTree

from swat.utils.compat import patch_pandas_sort
from swat.utils.testing import UUID_RE, get_cas_host_type, load_data

patch_pandas_sort()

USER, PASSWD = tm.get_user_pass()
HOST, PORT, PROTOCOL = tm.get_host_port_proto()


class TestUtils(tm.TestCase):

    server_type = None

    def setUp(self):
        swat.reset_option()
        swat.options.cas.print_messages = True
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

    def test_is_parallelizable(self):
        tbl = self.table
        with ResourceManager() as mgr:
            self.assertEqual(tbl.tableinfo().TableInfo.ix[0, 'Global'], 0)
            self.assertTrue(mgr.is_parallelizable(tbl) is False) 
            tbl.table.promote(drop=True)
            self.assertEqual(tbl.tableinfo().TableInfo.ix[0, 'Global'], 1)
            self.assertTrue(mgr.is_parallelizable(tbl) is True) 

    def test_emancipate(self):
        tbl = self.table
        with ResourceManager() as mgr:
            self.assertEqual(mgr.emancipate(), [])
            tbl.table.promote()
            tbl1, tbl2 = mgr.emancipate(tbl, tbl)             
            self.assertEqual(tbl1.name, tbl.name)
            self.assertEqual(tbl1.caslib, tbl.caslib)
            self.assertEqual(tbl2.name, tbl.name)
            self.assertEqual(tbl2.caslib, tbl.caslib)
            self.assertTrue(tbl1.get_connection() is not tbl.get_connection())
            self.assertTrue(tbl2.get_connection() is not tbl.get_connection())

    def test_unload_model(self):
        dtree = DecisionTree(target='Cylinders', inputs=['MSRP', 'Horsepower'])
        model = dtree.fit(self.table)
        self.assertEqual(model.data.table.tableexists().exists, 1)
        with ResourceManager() as mgr:
            mgr.track_model(model)
        self.assertEqual(model.data.table.tableexists().exists, 0)

    def test_unload_data(self):
        self.assertEqual(self.table.tableexists().exists, 1)
        with ResourceManager() as mgr:
            mgr.track_table(self.table)
        self.assertEqual(self.table.tableexists().exists, 0)
            
    def test_split_data_by_int(self):
        with ResourceManager() as mgr:
            out = mgr.split_data(self.table, k=3) 

            self.assertEqual(len(out), 3)
            self.assertTrue(isinstance(out[0], tuple))
            self.assertTrue(isinstance(out[1], tuple))
            self.assertTrue(isinstance(out[2], tuple))
            self.assertEqual(len(out[0]), 2)
            self.assertEqual(len(out[1]), 2)
            self.assertEqual(len(out[2]), 2)
            self.assertEqual(out[0][0].__class__.__name__, 'CASTable')
            self.assertEqual(out[0][1].__class__.__name__, 'CASTable')
            self.assertEqual(out[1][0].__class__.__name__, 'CASTable')
            self.assertEqual(out[1][1].__class__.__name__, 'CASTable')
            self.assertEqual(out[2][0].__class__.__name__, 'CASTable')
            self.assertEqual(out[2][1].__class__.__name__, 'CASTable')

            self.assertEqual(len(mgr.tables), 1)

            self.table.table.promote(drop=True)
        
            out = mgr.split_data(self.table, k=2)

            self.assertEqual(len(out), 2)
            self.assertTrue(isinstance(out[0], tuple))
            self.assertTrue(isinstance(out[1], tuple))
            self.assertEqual(len(out[0]), 2)
            self.assertEqual(len(out[1]), 2)
            self.assertEqual(out[0][0].__class__.__name__, 'CASTable')
            self.assertEqual(out[0][1].__class__.__name__, 'CASTable')
            self.assertEqual(out[1][0].__class__.__name__, 'CASTable')
            self.assertEqual(out[1][1].__class__.__name__, 'CASTable')

            self.assertEqual(len(mgr.tables), 2)

    def test_split_data_by_float(self):
        with ResourceManager() as mgr:
            out = mgr.split_data(self.table, k=0.3)

            self.assertEqual(len(out), 1)
            self.assertTrue(isinstance(out[0], tuple))
            self.assertEqual(len(out[0]), 2)
            self.assertEqual(out[0][0].__class__.__name__, 'CASTable')
            self.assertEqual(out[0][1].__class__.__name__, 'CASTable')

            self.assertEqual(len(mgr.tables), 1)

            self.table.table.promote(drop=True)

            out = mgr.split_data(self.table, k=0.4)

            self.assertEqual(len(out), 1)
            self.assertTrue(isinstance(out[0], tuple))
            self.assertEqual(len(out[0]), 2)
            self.assertEqual(out[0][0].__class__.__name__, 'CASTable')
            self.assertEqual(out[0][1].__class__.__name__, 'CASTable')

            self.assertEqual(len(mgr.tables), 2)


if __name__ == '__main__':
    tm.runtests()
