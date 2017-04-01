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
Tests for CAS Binning Transformer

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import numpy as np
import pandas as pd
import swat
import swat.utils.testing as tm
import unittest
from pipefitter.transformer import Binner

from swat.utils.compat import patch_pandas_sort
from swat.utils.testing import UUID_RE, get_cas_host_type, load_data

patch_pandas_sort()

USER, PASSWD = tm.get_user_pass()
HOST, PORT, PROTOCOL = tm.get_host_port_proto()


class TestBinner(tm.TestCase):

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
        self.s.endsession()
        del self.s
        swat.reset_option()

    def test_str(self):
        bin = Binner()
        self.assertEqual(str(bin).replace("u'", "'"), "Binner(method='bucket', n_bins=5)")

        bin = Binner('quantile')
        self.assertEqual(str(bin).replace("u'", "'"), "Binner(method='quantile', n_bins=5)")

        bin = Binner(method='quantile', n_bins=10)
        self.assertEqual(str(bin).replace("u'", "'"), "Binner(method='quantile', n_bins=10)")

        bin = Binner(method='quantile', n_bins=10, inputs='foo')
        self.assertEqual(str(bin).replace("u'", "'"), "Binner(method='quantile', n_bins=10, inputs=['foo'])")

    def test_repr(self):
        bin = Binner()
        self.assertEqual(repr(bin).replace("u'", "'"), "Binner(method='bucket', n_bins=5)")

    def test_params(self):
        tbl = self.table

        with self.assertRaises(ValueError):
            Binner(method='foo') 

        with self.assertRaises(ValueError):
            Binner(n_bins='foo') 

        bin = Binner(n_bins=0) 
        self.assertEqual(bin.n_bins, 1)

        bin = Binner(n_bins=2.5) 
        self.assertEqual(bin.n_bins, 2)

        bin = Binner(n_bins='10') 
        self.assertEqual(bin.n_bins, 10)

        with self.assertRaises(TypeError):
            Binner(foo='bar')

        bin = Binner(inputs='a')
        self.assertEqual(bin.inputs, ['a']) 

        bin = Binner(inputs=['a', 'b'])
        self.assertEqual(bin.inputs, ['a', 'b']) 

    def test_inputs(self):
        tbl = self.table

        bin = Binner(inputs=['MSRP', 'Weight'])
        out = bin.transform(tbl)

        self.assertEqual(len(out), 428)

        self.assertEqual(list(out.columns), ['Make', 'Model', 'Type', 'Origin',
                                             'DriveTrain', 'MSRP', 'Invoice',
                                             'EngineSize', 'Cylinders', 'Horsepower',
                                             'MPG_City', 'MPG_Highway',
                                             'Weight', 'Wheelbase', 'Length'])

        self.assertEqual(out.dtypes.to_dict(),
                         dict(Make='char', Model='char', Type='char',
                              Origin='char', DriveTrain='char', MSRP='double',
                              Invoice='double', EngineSize='double',
                              Cylinders='double', Horsepower='double',
                              MPG_City='double', MPG_Highway='double',
                              Weight='double', Wheelbase='double', Length='double'))

        self.assertEqual(out['MSRP'].value_counts().to_dict(),
                         {1.0: 366, 2.0: 51, 3.0: 7, 4.0: 3, 5.0: 1})
        self.assertTrue(len(out['Invoice'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['EngineSize'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['Cylinders'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['Horsepower'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['MPG_City'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['MPG_Highway'].value_counts().to_dict()) > 5)
        self.assertEqual(out['Weight'].value_counts().to_dict(),
                         {1.0: 78, 2.0: 245, 3.0: 87, 4.0: 15, 5.0: 3})
        self.assertTrue(len(out['Wheelbase'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['Length'].value_counts().to_dict()) > 5)

    def test_transform_params(self):
        tbl = self.table

        bin = Binner()

        # test inputs=

        self.assertTrue(bin.inputs is None)
        self.assertEqual(bin.n_bins, 5)

        out = bin.transform(tbl, inputs=['MSRP', 'Weight'])

        self.assertTrue(bin.inputs is None)
        self.assertEqual(len(out), 428)

        self.assertEqual(list(out.columns), ['Make', 'Model', 'Type', 'Origin',
                                             'DriveTrain', 'MSRP', 'Invoice',
                                             'EngineSize', 'Cylinders', 'Horsepower',
                                             'MPG_City', 'MPG_Highway',
                                             'Weight', 'Wheelbase', 'Length'])

        self.assertEqual(out.dtypes.to_dict(),
                         dict(Make='char', Model='char', Type='char',
                              Origin='char', DriveTrain='char', MSRP='double',
                              Invoice='double', EngineSize='double',
                              Cylinders='double', Horsepower='double',
                              MPG_City='double', MPG_Highway='double',
                              Weight='double', Wheelbase='double', Length='double'))

        self.assertEqual(out['MSRP'].value_counts().to_dict(),
                         {1.0: 366, 2.0: 51, 3.0: 7, 4.0: 3, 5.0: 1})
        self.assertTrue(len(out['Invoice'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['EngineSize'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['Cylinders'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['Horsepower'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['MPG_City'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['MPG_Highway'].value_counts().to_dict()) > 5)
        self.assertEqual(out['Weight'].value_counts().to_dict(),
                         {1.0: 78, 2.0: 245, 3.0: 87, 4.0: 15, 5.0: 3})
        self.assertTrue(len(out['Wheelbase'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['Length'].value_counts().to_dict()) > 5)

        # test n_bins=

        out = bin.transform(tbl, n_bins=3, inputs=['MSRP', 'Weight'])

        self.assertTrue(bin.inputs is None)
        self.assertEqual(bin.n_bins, 5)

        self.assertEqual(len(out), 428)

        self.assertEqual(list(out.columns), ['Make', 'Model', 'Type', 'Origin',
                                             'DriveTrain', 'MSRP', 'Invoice',
                                             'EngineSize', 'Cylinders', 'Horsepower',
                                             'MPG_City', 'MPG_Highway',
                                             'Weight', 'Wheelbase', 'Length'])

        self.assertEqual(out.dtypes.to_dict(),
                         dict(Make='char', Model='char', Type='char',
                              Origin='char', DriveTrain='char', MSRP='double',
                              Invoice='double', EngineSize='double',
                              Cylinders='double', Horsepower='double',
                              MPG_City='double', MPG_Highway='double',
                              Weight='double', Wheelbase='double', Length='double'))

        self.assertEqual(out['MSRP'].value_counts().to_dict(),
                         {1.0: 405, 2.0: 22, 3.0: 1})
        self.assertTrue(len(out['Invoice'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['EngineSize'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['Cylinders'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['Horsepower'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['MPG_City'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['MPG_Highway'].value_counts().to_dict()) > 5)
        self.assertEqual(out['Weight'].value_counts().to_dict(),
                         {1.0: 252, 2.0: 166, 3.0: 10})
        self.assertTrue(len(out['Wheelbase'].value_counts().to_dict()) > 5)
        self.assertTrue(len(out['Length'].value_counts().to_dict()) > 5)

    def test_basic_binning(self):
        tbl = self.table

        # default parameters

        bin = Binner()
        out = bin.transform(tbl)

        self.assertEqual(len(out), 428)

        self.assertEqual(list(out.columns), ['Make', 'Model', 'Type', 'Origin',
                                             'DriveTrain', 'MSRP', 'Invoice',
                                             'EngineSize', 'Cylinders', 'Horsepower',
                                             'MPG_City', 'MPG_Highway',
                                             'Weight', 'Wheelbase', 'Length'])

        self.assertEqual(out.dtypes.to_dict(),
                         dict(Make='char', Model='char', Type='char',
                              Origin='char', DriveTrain='char', MSRP='double',
                              Invoice='double', EngineSize='double',
                              Cylinders='double', Horsepower='double',
                              MPG_City='double', MPG_Highway='double',
                              Weight='double', Wheelbase='double', Length='double'))

        self.assertEqual(out['MSRP'].value_counts().to_dict(), 
                         {1.0: 366, 2.0: 51, 3.0: 7, 4.0: 3, 5.0: 1})
        self.assertEqual(out['Invoice'].value_counts().to_dict(),
                         {1.0: 366, 2.0: 52, 3.0: 6, 4.0: 3, 5.0: 1})
        self.assertEqual(out['EngineSize'].value_counts().to_dict(),
                         {1.0: 159, 2.0: 173, 3.0: 80, 4.0: 15, 5.0: 1})
        self.assertEqual(out['Cylinders'].value_counts().to_dict(),
                         {1.0: 137, 2.0: 197, 3.0: 87, 4.0: 2, 5.0: 3})
        self.assertEqual(out['Horsepower'].value_counts().to_dict(),
                         {1.0: 92, 2.0: 218, 3.0: 92, 4.0: 19, 5.0: 7})
        self.assertEqual(out['MPG_City'].value_counts().to_dict(),
                         {1.0: 226, 2.0: 186, 3.0: 13, 4.0: 1, 5.0: 2})
        self.assertEqual(out['MPG_Highway'].value_counts().to_dict(),
                         {1.0: 83, 2.0: 302, 3.0: 39, 4.0: 3, 5.0: 1})
        self.assertEqual(out['Weight'].value_counts().to_dict(),
                         {1.0: 78, 2.0: 245, 3.0: 87, 4.0: 15, 5.0: 3})
        self.assertEqual(out['Wheelbase'].value_counts().to_dict(),
                         {1.0: 52, 2.0: 232, 3.0: 118, 4.0: 20, 5.0: 6})
        self.assertEqual(out['Length'].value_counts().to_dict(),
                         {1.0: 20, 2.0: 135, 3.0: 198, 4.0: 64, 5.0: 11})

        # non-default parameters

        bin = Binner(method='quantile', n_bins=4)
        out = bin.transform(tbl)

        self.assertEqual(len(out), 428)

        self.assertEqual(list(out.columns), ['Make', 'Model', 'Type', 'Origin',
                                             'DriveTrain', 'MSRP', 'Invoice',
                                             'EngineSize', 'Cylinders', 'Horsepower',
                                             'MPG_City', 'MPG_Highway',
                                             'Weight', 'Wheelbase', 'Length'])

        self.assertEqual(out.dtypes.to_dict(),
                         dict(Make='char', Model='char', Type='char',
                              Origin='char', DriveTrain='char', MSRP='double',
                              Invoice='double', EngineSize='double',
                              Cylinders='double', Horsepower='double',
                              MPG_City='double', MPG_Highway='double',
                              Weight='double', Wheelbase='double', Length='double'))

        self.assertEqual(out['MSRP'].value_counts().to_dict(),
                         {1.0: 107, 2.0: 107, 3.0: 107, 4.0: 107}) 
        self.assertEqual(out['Invoice'].value_counts().to_dict(),
                         {1.0: 107, 2.0: 107, 3.0: 107, 4.0: 107})
        self.assertEqual(out['EngineSize'].value_counts().to_dict(),
                         {1.0: 107, 2.0: 71, 3.0: 142, 4.0: 108})
        self.assertEqual(out['Cylinders'].value_counts().to_dict(),
                         {1.0: 1, 2.0: 143, 4.0: 282})
        self.assertEqual(out['Horsepower'].value_counts().to_dict(),
                         {1.0: 104, 2.0: 106, 3.0: 110, 4.0: 108})
        self.assertEqual(out['MPG_City'].value_counts().to_dict(),
                         {1.0: 79, 2.0: 110, 3.0: 132, 4.0: 107})
        self.assertEqual(out['MPG_Highway'].value_counts().to_dict(),
                         {1.0: 99, 2.0: 69, 3.0: 120, 4.0: 140})
        self.assertEqual(out['Weight'].value_counts().to_dict(),
                         {1.0: 107, 2.0: 107, 3.0: 107, 4.0: 107})
        self.assertEqual(out['Wheelbase'].value_counts().to_dict(),
                         {1.0: 91, 2.0: 102, 3.0: 108, 4.0: 127})
        self.assertEqual(out['Length'].value_counts().to_dict(),
                         {1.0: 104, 2.0: 106, 3.0: 105, 4.0: 113})

if __name__ == '__main__':
    tm.runtests()
