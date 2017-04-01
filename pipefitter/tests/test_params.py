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
Tests for parameter management

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import functools
import os
import re
import six
import swat.utils.testing as tm
import unittest
from pipefitter.utils.params import (ParameterManager, check_int, check_float,
                                     check_string, check_url, check_boolean, check_url,
                                     check_variable, check_variable_list, check_number,
                                     check_int_list, check_float_list, check_number_list,
                                     Parameter, param_property, param_def,
                                     READ_ONLY_PARAMETER)


class TestParamProperty(tm.TestCase):

    def test_get(self):
        pp = param_property('int_param', 'Integer parameter')
        self.assertEqual(pp.__get__(None), pp)


class TestParameter(tm.TestCase):

    def test_str(self):
        class MyPM(ParameterManager):
            param_defs = dict(
                int_param=param_def(1, check_int, 'Integer parameter'),
                float_param=param_def(12.34, check_float, 'Float parameter'),
                str_param=param_def('some text', check_string, 'String parameter'),
            )

        pm = MyPM()

        self.assertEqual(str(pm.int_param), '1') 
        self.assertEqual(str(pm.float_param), '12.34') 
        self.assertEqual(str(pm.str_param), 'some text') 

        self.assertEqual(repr(pm.int_param), '1') 
        self.assertEqual(repr(pm.float_param), '12.34') 
        self.assertEqual(repr(pm.str_param).replace("u'", "'"), "'some text'") 

    def test_get_default(self):
        class MyPM(ParameterManager):
            param_defs = dict(
                int_param=param_def(1, check_int, 'Integer parameter'),
                float_param=param_def(12.34, check_float, 'Float parameter'),
                str_param=param_def('some text', check_string, 'String parameter'),
            )

        pm = MyPM()

        self.assertEqual(pm.int_param.get_default(), 1)
        self.assertTrue(pm.int_param.is_default())

        pm.int_param = 100

        self.assertEqual(pm.int_param.get_default(), 1)
        self.assertFalse(pm.int_param.is_default())


class TestParameterManager(tm.TestCase):

    def test_init(self):
        class MyPM(ParameterManager):
            param_defs = dict(
                int_param=Parameter(None, 'int_param', 2)
            )

        pm = MyPM()

        self.assertTrue('int_param' in pm.params)
        self.assertEqual(pm.int_param, 2)

    def test_set_options(self):
        class MyPM(ParameterManager):
            param_defs = dict(
                int_param=param_def(1, check_int, 'Integer parameter'),
                float_param=param_def(12.34, check_float, 'Float parameter'),
                str_param=param_def('some text', check_string, 'String parameter'),
            )

        pm = MyPM()
        
        self.assertEqual(pm.int_param._options, 0)
        self.assertEqual(pm.float_param._options, 0)
        self.assertEqual(pm.str_param._options, 0)

        pm._set_options(READ_ONLY_PARAMETER)

        self.assertEqual(pm.int_param._options, READ_ONLY_PARAMETER)
        self.assertEqual(pm.float_param._options, READ_ONLY_PARAMETER)
        self.assertEqual(pm.str_param._options, READ_ONLY_PARAMETER)

    def test_str(self):
        class MyPM(ParameterManager):
            param_defs = dict(
                int_param=param_def(1, check_int, 'Integer parameter'),
                float_param=param_def(12.34, check_float, 'Float parameter'),
                str_param=param_def('some text', check_string, 'String parameter'),
            )

        pm = MyPM()

        self.assertEqual(str(pm).replace("u'", "'"),
                         'MyPM(float_param=12.34, int_param=1, '
                         'str_param=\'some text\')')

        self.assertEqual(repr(pm).replace("u'", "'"),
                         'MyPM(float_param=12.34, int_param=1, '
                         'str_param=\'some text\')')


class TestParameterDict(tm.TestCase):

    def setUp(self):
        class MyPM(ParameterManager):
            param_defs = dict(
                int_param=param_def(1, check_int, 'Integer parameter'),
                float_param=param_def(12.34, check_float, 'Float parameter'),
                str_param=param_def('some text', check_string, 'String parameter'),
            )
        self.pm = MyPM()
        self.pd = self.pm.params

    def tearDown(self):
        pass

    def test_copy(self):
        import copy

        pd2 = copy.copy(self.pd)

        self.assertEqual(sorted(self.pd.items()), sorted(pd2.items())) 

    def test_deepcopy(self):
        import copy

        pd2 = copy.deepcopy(self.pd)

        self.assertEqual(sorted(self.pd.items()), sorted(pd2.items())) 

    def test_getitem(self):
        self.assertTrue(self.pd[self.pd.get_parameter('int_param')] is self.pd['int_param'])
        self.assertEqual(self.pd[self.pd.get_parameter('int_param')], 1)
        self.assertEqual(self.pd['int_param'], 1)

    def test_setitem(self):
        class MyPM(ParameterManager):
            param_defs = dict(
                int_param=param_def(1, check_int, 'Integer parameter',
                                    options=READ_ONLY_PARAMETER),
                float_param=param_def(12.34, check_float, 'Float parameter'),
            )

        pm = MyPM()
        pd = pm.params

        self.assertEqual(pd['int_param'], 1)
        self.assertEqual(pd['float_param'], 12.34)
  
        with self.assertRaises(RuntimeError):
            pd['int_param'] = 20

        pd['float_param'] = 0.2
        self.assertEqual(pd['float_param'], 0.2)

    def test_add_parameter(self):
        self.pd.add_parameter(Parameter(self.pm, 'foo', 100))
        self.assertEqual(self.pd['foo'], 100)

        with self.assertRaises(TypeError):
            self.pd.add_parameter('foo')

    def test_get_filtered_params(self):
        pm1 = self.pm

        class MyPM2(ParameterManager):
            param_defs = dict(
                int_param=param_def(2, check_int, 'Integer parameter'),
                url_param=param_def('http://www.sas.com', check_url, 'URL parameter'),
            )

        pm2 = MyPM2()

        # Parameter objects as keys

        params = {pm1.int_param: 100, pm1.str_param: 'more text',
                  pm2.int_param: 999}

        self.assertEqual(pm1.get_filtered_params(params),
                         {'int_param': 100, 'float_param': 12.34, 'str_param': 'more text'})
        self.assertEqual(pm1.params,
                         {'int_param': 1, 'float_param': 12.34, 'str_param': 'some text'})
        self.assertEqual(pm2.get_filtered_params(params),
                         {'int_param': 999, 'url_param': 'http://www.sas.com'})
        self.assertEqual(pm2.params,
                         {'int_param': 2, 'url_param': 'http://www.sas.com'})

        # Strings as keys

        params = {'int_param': 12345, 'float_param': 6.66}

        self.assertEqual(pm1.get_filtered_params(params),
                         {'int_param': 12345, 'float_param': 6.66, 'str_param': 'some text'})
        self.assertEqual(pm1.params,
                         {'int_param': 1, 'float_param': 12.34, 'str_param': 'some text'})
        self.assertEqual(pm2.get_filtered_params(params),
                         {'int_param': 12345, 'url_param': 'http://www.sas.com'})
        self.assertEqual(pm2.params,
                         {'int_param': 2, 'url_param': 'http://www.sas.com'})

        # ParameterManager object arguments

        self.assertEqual(pm1.get_filtered_params(pm2),
                         {'int_param': 1, 'float_param': 12.34, 'str_param': 'some text'})
        self.assertEqual(pm1.params,
                         {'int_param': 1, 'float_param': 12.34, 'str_param': 'some text'})
        self.assertEqual(pm2.params,
                         {'int_param': 2, 'url_param': 'http://www.sas.com'})

        # Parameter object arguments

        args = (pm1.int_param, 100, pm2.int_param, 300)

        self.assertEqual(pm1.get_filtered_params(*args),
                         {'int_param': 100, 'float_param': 12.34, 'str_param': 'some text'})
        self.assertEqual(pm2.get_filtered_params(*args),
                         {'int_param': 300, 'url_param': 'http://www.sas.com'})
        self.assertEqual(pm1.params,
                         {'int_param': 1, 'float_param': 12.34, 'str_param': 'some text'})
        self.assertEqual(pm2.params,
                         {'int_param': 2, 'url_param': 'http://www.sas.com'})

        # Tuple arguments

        with self.assertRaises(ValueError):
            pm1.get_filtered_params(('int_param',))

        with self.assertRaises(ValueError):
            pm1.get_filtered_params(('int_param', 100, 200))

        args = ((pm1.int_param, 200), ('float_param', 99.9), (pm2.int_param, 300))

        self.assertEqual(pm1.get_filtered_params(*args),
                         {'int_param': 200, 'float_param': 99.9, 'str_param': 'some text'})
        self.assertEqual(pm2.get_filtered_params(*args),
                         {'int_param': 300, 'url_param': 'http://www.sas.com'})
        self.assertEqual(pm1.params,
                         {'int_param': 1, 'float_param': 12.34, 'str_param': 'some text'})
        self.assertEqual(pm2.params,
                         {'int_param': 2, 'url_param': 'http://www.sas.com'})

        # Consecutive arguments

        with self.assertRaises(ValueError):
            pm1.get_filtered_params('int_param')

        with self.assertRaises(TypeError):
            pm1.get_filtered_params(10)
        
        args = ('int_param', 200, 'float_param', 99.9)

        self.assertEqual(pm1.get_filtered_params(*args),
                         {'int_param': 200, 'float_param': 99.9, 'str_param': 'some text'})
        self.assertEqual(pm1.params,
                         {'int_param': 1, 'float_param': 12.34, 'str_param': 'some text'})

    def test_get_combined_params(self):
        pm1 = self.pm

        class MyPM2(ParameterManager):
            param_defs = dict(
                int_param=param_def(2, check_int, 'Integer parameter'),
                url_param=param_def('http://www.sas.com', check_url, 'URL parameter'),
            )

        pm2 = MyPM2()

        # Parameter objects as keys

        params = {pm1.int_param: 100, pm1.str_param: 'more text'}

        self.assertEqual(pm1.get_combined_params(params),
                         {'int_param': 100, 'float_param': 12.34, 'str_param': 'more text'})
        self.assertEqual(pm1.params,
                         {'int_param': 1, 'float_param': 12.34, 'str_param': 'some text'})
        with self.assertRaises(KeyError):
            pm2.get_combined_params(params),

        # Strings as keys

        params = {'int_param': 12345, 'float_param': 6.66}

        self.assertEqual(pm1.get_combined_params(params),
                         {'int_param': 12345, 'float_param': 6.66, 'str_param': 'some text'})
        self.assertEqual(pm1.params,
                         {'int_param': 1, 'float_param': 12.34, 'str_param': 'some text'})
        with self.assertRaises(KeyError):
            pm2.get_combined_params(params)

        # ParameterManager object arguments

        with self.assertRaises(TypeError):
            pm1.get_combined_params(pm2)

        # Parameter object arguments

        args = (pm1.int_param, 100)

        self.assertEqual(pm1.get_combined_params(*args),
                         {'int_param': 100, 'float_param': 12.34, 'str_param': 'some text'})
        self.assertEqual(pm2.get_combined_params(*args),
                         {'int_param': 100, 'url_param': 'http://www.sas.com'})
        self.assertEqual(pm1.params,
                         {'int_param': 1, 'float_param': 12.34, 'str_param': 'some text'})
        self.assertEqual(pm2.params,
                         {'int_param': 2, 'url_param': 'http://www.sas.com'})

        # Tuple arguments

        with self.assertRaises(ValueError):
            pm1.get_combined_params(('int_param',))

        with self.assertRaises(ValueError):
            pm1.get_combined_params(('int_param', 100, 200))

        args = ((pm1.int_param, 200), ('float_param', 99.9))

        self.assertEqual(pm1.get_combined_params(*args),
                         {'int_param': 200, 'float_param': 99.9, 'str_param': 'some text'})
        self.assertEqual(pm1.params,
                         {'int_param': 1, 'float_param': 12.34, 'str_param': 'some text'})
        with self.assertRaises(KeyError):
            pm2.get_combined_params(*args)

        # Bad tuple argument

        args = ((100, 200),)

        with self.assertRaises(TypeError):
            pm1.get_combined_params(*args)

        # Consecutive arguments

        with self.assertRaises(ValueError):
            pm1.get_combined_params('int_param')

        with self.assertRaises(TypeError):
            pm1.get_combined_params(10)

        args = ('int_param', 200, 'float_param', 99.9)

        self.assertEqual(pm1.get_combined_params(*args),
                         {'int_param': 200, 'float_param': 99.9, 'str_param': 'some text'})
        self.assertEqual(pm1.params,
                         {'int_param': 1, 'float_param': 12.34, 'str_param': 'some text'})
        
    def test_str(self):
        self.assertEqual(self.pd, eval(str(self.pd)))

    def test_repr(self):
        self.assertEqual(self.pd, eval(str(self.pd)))

    def test_del_parameter(self):
        pd = self.pd

        self.assertEqual(pd, {'int_param': 1, 'float_param': 12.34, 'str_param': 'some text'})

        pd.del_parameter('str_param')
        self.assertEqual(pd, {'int_param': 1, 'float_param': 12.34})

        pd.del_parameter('int_param', 'float_param')
        self.assertEqual(pd, {})

        pd.del_parameter('str_param')
        self.assertEqual(pd, {})

    def test_get_parameter(self):
        pd = self.pd

        self.assertEqual(pd.get_parameter('int_param'), 1)
        self.assertTrue(isinstance(pd.get_parameter('int_param'), Parameter))

        with self.assertRaises(KeyError):
            pd.get_parameter('foo')

        self.assertEqual(pd.get_parameter('foo', 100), 100)

        # Get using a Parameter key
        self.assertEqual(pd.get_parameter(self.pm.int_param), 1)

    def test_get(self):
        pd = self.pd

        self.assertEqual(pd.get('int_param'), 1)
        self.assertTrue(isinstance(pd.get('int_param'), six.integer_types))

        with self.assertRaises(KeyError):
            pd.get('foo')

        self.assertEqual(pd.get('foo', 100), 100)

        # Get using a Parameter key
        self.assertEqual(pd.get(self.pm.int_param), 1)

    def test_describe_parameter(self):
        pd = self.pd
        output = six.StringIO()

        pd.describe_parameter('int_param', output=output)
        output.seek(0)
        self.assertEqual(output.read(),
            'int_param\n    Integer parameter\n    [Current: 1] [Default: 1]')

        output = six.StringIO()

        pd.describe_parameter('int_param', 'str_param', output=output)
        output.seek(0)
        self.assertEqual(output.read().replace("u'", "'"),
            'int_param\n    Integer parameter\n    [Current: 1] [Default: 1]\n\n'
            'str_param\n    String parameter\n    [Current: \'some text\'] [Default: \'some text\']')

        with self.assertRaises(KeyError):
            pd.describe_parameter('foo')

    def test_len(self):
        self.assertEqual(len(self.pd), 3)

    def test_iter(self):
        keys = []
        for key in self.pd:
            keys.append(key)
        self.assertEqual(sorted(keys), ['float_param', 'int_param', 'str_param'])

    def test_update(self):
        pd = self.pd

        self.assertEqual(pd, {'int_param': 1, 'float_param': 12.34, 'str_param': 'some text'})
        
        pd.update({self.pm.int_param: 200, self.pm.str_param: 'hi'})
        self.assertEqual(pd, {'int_param': 200, 'float_param': 12.34, 'str_param': 'hi'})

        class MyPM2(ParameterManager):
            param_defs = dict(
                float_param=param_def(99.99, check_float, 'Float parameter'),
            )

        pm2 = MyPM2()

        pd.update(pm2.float_param)
        self.assertEqual(pd, {'int_param': 200, 'float_param': 99.99, 'str_param': 'hi'})

        pd.update(('int_param', 5))
        self.assertEqual(pd, {'int_param': 5, 'float_param': 99.99, 'str_param': 'hi'})


class TestParameters(tm.TestCase):

    def setUp(self):
        class MyPM(ParameterManager):
            param_defs = dict(
                int_param=param_def(1, check_int, 'Integer parameter'),
                float_param=param_def(12.34, check_float, 'Float parameter'),
                str_param=param_def('some text', check_string, 'String parameter'),
            )

        self.pm = MyPM()

    def tearDown(self):
        pass

    def test_values(self):
        pm = self.pm

        self.assertEqual(set(pm.params.keys()), set(['int_param', 'float_param', 'str_param']))
        self.assertEqual(set(pm.params.values()), set([1, 12.34, 'some text']))

        self.assertEqual(pm.params, {'int_param': 1, 'float_param': 12.34,
                                      'str_param': 'some text'}) 

        self.assertEqual(pm.params, {pm.int_param: 1, pm.float_param: 12.34,
                                     pm.str_param: 'some text'}) 

        class MyPM2(ParameterManager):
            param_defs = dict(
                int_param=param_def(1, check_int, 'Integer parameter'),
                float_param=param_def(12.34, check_float, 'Float parameter'),
                str_param=param_def('some text', check_string, 'String parameter'),
            )

        pm2 = MyPM2()

        self.assertEqual(pm2.params, pm.params)

    def test_properties(self):
        pm = self.pm

        pm.int_param = 10
        self.assertEqual(pm.int_param, 10)
        self.assertTrue(isinstance(pm.int_param, Parameter))

        with self.assertRaises(AttributeError):
            pm.foo_param

    def test_check_int(self):
        class IntParam(ParameterManager):
            param_defs = dict(
                int_param=param_def(1, functools.partial(check_int, minimum=0, maximum=100),
                                   'Integer parameter'),
            )

        pm = IntParam()

        # Incorrect types

        with self.assertRaises(ValueError):
            pm.int_param = 'foo'

        # Range values for integers

        pm.int_param = 50
        self.assertEqual(pm.int_param, 50)

        pm.int_param = 0
        self.assertEqual(pm.int_param, 0)

        pm.int_param = 100
        self.assertEqual(pm.int_param, 100)

        with self.assertRaises(ValueError):
            pm.int_param = -1

        with self.assertRaises(ValueError):
            pm.int_param = 101

        class IntParam2(ParameterManager):
            param_defs = dict(
                int_param=param_def(10, functools.partial(check_int, multiple_of=5),
                                   'Integer parameter'),
            )

        pm = IntParam2()

        pm.int_param = 20
        self.assertEqual(pm.int_param, 20)

        with self.assertRaises(ValueError):
            pm.int_param = 23

        # Exclusive min / max

        class IntParam3(ParameterManager):
            param_defs = dict(
                int_param=param_def(1, functools.partial(check_int, minimum=0,
                                                         maximum=100,
                                                         exclusive_minimum=True,
                                                         exclusive_maximum=True),
                                   'Integer parameter'),
            )

        pm = IntParam3()
        
        pm.int_param = 1
        self.assertEqual(pm.int_param, 1)

        pm.int_param = 99
        self.assertEqual(pm.int_param, 99)

        with self.assertRaises(ValueError):
            pm.int_param = 0

        with self.assertRaises(ValueError):
            pm.int_param = 100

        # None value

        with self.assertRaises(TypeError):
            pm.int_param = None

        class IntParam4(ParameterManager):
            param_defs = dict(
                int_param=param_def(1, functools.partial(check_int, minimum=0,
                                                         maximum=100,
                                                         allow_none=True,
                                                         exclusive_minimum=True,
                                                         exclusive_maximum=True),
                                   'Integer parameter'),
            )

        pm = IntParam4()

        pm.int_param = None
        self.assertEqual(pm.int_param, None)

    def test_check_float(self):
        class FloatPM(ParameterManager):
            param_defs = dict(
                float_param=param_def(12.34,
                                      functools.partial(check_float, minimum=1, maximum=99),
                                     'Float parameter'),
            )

        pm = FloatPM()

        # Incorrect types

        with self.assertRaises(ValueError):
            pm.float_param = 'foo'

        # Range values for floats

        pm.float_param = 50.0
        self.assertEqual(pm.float_param, 50)

        pm.float_param = 1.0
        self.assertEqual(pm.float_param, 1)

        pm.float_param = 99.0
        self.assertEqual(pm.float_param, 99)

        with self.assertRaises(ValueError):
            pm.float_param = 0.9

        with self.assertRaises(ValueError):
            pm.float_param = 99.1

        class FloatPM2(ParameterManager):
            param_defs = dict(
                float_param=param_def(12.34, functools.partial(check_float,
                                                     minimum=1, exclusive_minimum=True,
                                                     maximum=99, exclusive_maximum=True),
                                     'Float parameter'),
            )

        pm = FloatPM2()

        with self.assertRaises(ValueError):
            pm.float_param = 1

        with self.assertRaises(ValueError):
            pm.float_param = 99 

        # Multiples

        class FloatPM3(ParameterManager):
            param_defs = dict(
                float_param=param_def(10.0, functools.partial(check_float, multiple_of=5),
                                     'Float parameter'),
            )

        pm = FloatPM3()

        pm.float_param = 25.0
        self.assertEqual(pm.float_param, 25)

        with self.assertRaises(ValueError):
            pm.float_param = 23.0

        # None value

        with self.assertRaises(TypeError):
            pm.float_param = None

        class FloatPM4(ParameterManager):
            param_defs = dict(
                float_param=param_def(10.0, functools.partial(check_float, allow_none=True),
                                     'Float parameter'),
            )

        pm = FloatPM4()

        pm.float_param = None
        self.assertEqual(pm.float_param, None)

    def test_check_number(self):
        class NumParam(ParameterManager):
            param_defs = dict(
                num_param=param_def(1, functools.partial(check_number, minimum=0, maximum=100),
                                   'Number parameter'),
            )

        pm = NumParam()

        # Incorrect types

        with self.assertRaises(ValueError):
            pm.num_param = 'foo'

        # Range values for integers

        pm.num_param = 50
        self.assertEqual(pm.num_param, 50)

        pm.num_param = 0
        self.assertEqual(pm.num_param, 0)

        pm.num_param = 100
        self.assertEqual(pm.num_param, 100)

        with self.assertRaises(ValueError):
            pm.num_param = -1

        with self.assertRaises(ValueError):
            pm.num_param = 101

        # Values for floats

        pm.num_param = 12.34
        self.assertEqual(pm.num_param, 12.34)

        # Multiples

        class NumParam2(ParameterManager):
            param_defs = dict(
                num_param=param_def(10, functools.partial(check_number, multiple_of=5),
                                   'Number parameter'),
            )

        pm = NumParam2()

        pm.num_param = 20
        self.assertEqual(pm.num_param, 20)

        with self.assertRaises(ValueError):
            pm.num_param = 23

        # Exclusive min / max

        class NumParam3(ParameterManager):
            param_defs = dict(
                num_param=param_def(1, functools.partial(check_number, minimum=0,
                                                         maximum=100,
                                                         exclusive_minimum=True,
                                                         exclusive_maximum=True),
                                   'Number parameter'),
            )

        pm = NumParam3()

        pm.num_param = 1
        self.assertEqual(pm.num_param, 1)

        pm.num_param = 99
        self.assertEqual(pm.num_param, 99)

        with self.assertRaises(ValueError):
            pm.num_param = 0

        with self.assertRaises(ValueError):
            pm.num_param = 100

        # None value

        with self.assertRaises(TypeError):
            pm.num_param = None

        class NumParam4(ParameterManager):
            param_defs = dict(
                num_param=param_def(1, functools.partial(check_number, minimum=0,
                                                         maximum=100,
                                                         allow_none=True,
                                                         exclusive_minimum=True,
                                                         exclusive_maximum=True),
                                   'Number parameter'),
            )

        pm = NumParam4()

        pm.num_param = 1
        self.assertEqual(pm.num_param, 1)

        pm.num_param = None
        self.assertEqual(pm.num_param, None)

    def test_check_string(self):
        class StringPM(ParameterManager):
            param_defs = dict(
                str_param=param_def('some text', 
                                    functools.partial(check_string, pattern=r' +'),
                                    'String parameter'),
                select_param=param_def('one', 
                                       functools.partial(check_string,
                                                         valid_values=['one', 'two', 'three'],
                                                         normalize=True),
                                       'Select parameter'),
                regex_param=param_def('abc1',
                                      functools.partial(check_string, pattern=re.compile(r'\d+')),
                                      'Regex parameter'),
            )

        pm = StringPM()

        # Incorrect types

        with self.assertRaises(TypeError):
            pm.str_param = 100

        # String patterns

        pm.str_param = 'even more text'
        self.assertEqual(pm.str_param, 'even more text')

        with self.assertRaises(ValueError):
            pm.str_param = 'foo'

        pm.regex_param = 'even 2 text'
        self.assertEqual(pm.regex_param, 'even 2 text')

        with self.assertRaises(ValueError):
            pm.regex_param = 'even text'

        # String selection

        pm.select_param = 'two'
        self.assertEqual(pm.select_param, 'two')

        pm.select_param = 'three'
        self.assertEqual(pm.select_param, 'three')

        pm.select_param = 'TwO'
        self.assertEqual(pm.select_param, 'two')

        with self.assertRaises(ValueError):
            pm.select_param = 'foo'

        # Min / max lengths

        class StringPM2(ParameterManager):
            param_defs = dict(
                str_param=param_def('text',
                                    functools.partial(check_string, min_length=2, max_length=4),
                                    'String parameter'),
            )

        pm = StringPM2()

        with self.assertRaises(ValueError):
            pm.str_param = 'foobar'

        with self.assertRaises(ValueError):
            pm.str_param = 'f'

        pm.str_param = 'foo'
        self.assertEqual(pm.str_param, 'foo')

    def test_check_variable(self):
        class VarPM(ParameterManager):
            param_defs = dict(
                var_param=param_def('varname',
                                    functools.partial(check_variable, pattern=r'[A-Za-z]\w*',
                                                      allow_none=False))
            )

        pm = VarPM()

        pm.var_param = 'foo'
        self.assertEqual(pm.var_param, 'foo')

        with self.assertRaises(ValueError):
            pm.var_param = '6'
        self.assertEqual(pm.var_param, 'foo')

        with self.assertRaises(TypeError):
            pm.var_param = 6
        self.assertEqual(pm.var_param, 'foo')

        with self.assertRaises(TypeError):
            pm.var_param = None
        self.assertEqual(pm.var_param, 'foo')

    def test_check_variable_list(self):
        class VarPM(ParameterManager):
            param_defs = dict(
                var_param=param_def('varname',
                                    functools.partial(check_variable_list,
                                                      pattern=r'[A-Za-z]\w*',
                                                      allow_empty=False))
            )

        pm = VarPM()

        pm.var_param = 'foo'
        self.assertEqual(pm.var_param, ['foo'])

        pm.var_param = ['one', 'two']
        self.assertEqual(pm.var_param, ['one', 'two'])

        with self.assertRaises(ValueError):
            pm.var_param = '6'
        self.assertEqual(pm.var_param, ['one', 'two'])

        with self.assertRaises(TypeError):
            pm.var_param = 6
        self.assertEqual(pm.var_param, ['one', 'two'])

        with self.assertRaises(ValueError):
            pm.var_param = None
        self.assertEqual(pm.var_param, ['one', 'two'])

        with self.assertRaises(ValueError):
            pm.var_param = []
        self.assertEqual(pm.var_param, ['one', 'two'])

    def test_check_boolean(self):
        class BoolPM(ParameterManager):
            param_defs = dict(
                bool_param=param_def(True, check_boolean, 'Boolean parameter'),
            )

        pm = BoolPM()

        with self.assertRaises(TypeError):
            pm.bool_param = 'foo'

        pm.bool_param = False
        self.assertEqual(pm.bool_param, False)

        pm.bool_param = 1
        self.assertEqual(pm.bool_param, True)

        pm.bool_param = 0
        self.assertEqual(pm.bool_param, False)

        with self.assertRaises(ValueError):
            pm.bool_param = 100

    def test_check_url(self):
        class URLPM(ParameterManager):
            param_defs = dict(
                url_param=param_def('http://www.sas.com', check_url, 'URL parameter'),
            )

        pm = URLPM()

        with self.assertRaises(TypeError):
            pm.url_param = 100

        pm.url_param = 'http://www.google.com'
        self.assertEqual(pm.url_param, 'http://www.google.com')

# It seems impossible to error with an invalid URL
#       with self.assertRaises(ValueError):
#           pm.url_param = 'http://::1/'

    def test_update(self):
        pm = self.pm

        pm.params.update({pm.int_param: 20, pm.float_param: 999.99})
        self.assertEqual(pm.params, {'int_param': 20, 'float_param': 999.99,
                                      'str_param': 'some text'}) 

        pm.params.update({'int_param': 55, 'float_param': 1.896})
        self.assertEqual(pm.params, {'int_param': 55, 'float_param': 1.896,
                                      'str_param': 'some text'}) 

    def test_param_ids(self):
        self.assertEqual(len(Parameter.param_ids), 3)
        self.assertEqual(Parameter.param_ids['int_param'], '10000')
        self.assertEqual(Parameter.param_ids['float_param'], '10001')
        self.assertEqual(Parameter.param_ids['str_param'], '10002')

        pm2 = ParameterManager(
            ('int_param', 1000, check_int, 'Integer parameter'),
            ('num_param', 2000, check_float, 'Numeric parameter'),
        )

        self.assertEqual(len(Parameter.param_ids), 4)
        self.assertEqual(Parameter.param_ids['int_param'], '10000')
        self.assertEqual(Parameter.param_ids['float_param'], '10001')
        self.assertEqual(Parameter.param_ids['str_param'], '10002')
        self.assertEqual(Parameter.param_ids['num_param'], '10003')

    def test_update(self):
        pm = self.pm

        pm.params.update({pm.int_param: 20, pm.float_param: 999.99})
        self.assertEqual(pm.params, {'int_param': 20, 'float_param': 999.99,
                                      'str_param': 'some text'}) 

        pm.params.update({'int_param': 55, 'float_param': 1.896})
        self.assertEqual(pm.params, {'int_param': 55, 'float_param': 1.896,
                                      'str_param': 'some text'}) 

    def test_param_ids(self):
#       self.assertEqual(Parameter.param_ids['int_param'], '10000')
#       self.assertEqual(Parameter.param_ids['float_param'], '10001')
#       self.assertEqual(Parameter.param_ids['str_param'], '10002')

        int_param_id = Parameter.param_ids['int_param']
        max_id = max(int(x) for x in Parameter.param_ids.values())

        class MyPM2(ParameterManager):
            param_defs = dict(
                int_param=param_def(1000, check_int, 'Integer parameter'),
                foo_param=param_def(2000, check_float, 'Numeric parameter'),
            )

        pm2 = MyPM2()

        self.assertEqual(Parameter.param_ids['int_param'], int_param_id)
#       self.assertEqual(Parameter.param_ids['float_param'], '10001')
#       self.assertEqual(Parameter.param_ids['str_param'], '10002')
        self.assertEqual(Parameter.param_ids['foo_param'], str(max_id + 1))

    def test_comparisons(self):
        pm = self.pm

        class MyPM2(ParameterManager):
            param_defs = dict(
                int_one=param_def(100, check_int, 'Integer parameter'),
                int_two=param_def(200, check_int, 'Integer parameter'),
                int_minus_one=param_def(99, check_int, 'Integer parameter'),
                int_plus_one=param_def(101, check_int, 'Integer parameter'),
            )

        pm2 = MyPM2()

        pm.int_param = 100
        
        self.assertTrue(pm.int_param == 100)
        self.assertTrue(pm.int_param == pm2.int_one)
        self.assertFalse(pm.int_param == pm2.int_two)

        self.assertFalse(pm.int_param != 100)
        self.assertFalse(pm.int_param != pm2.int_one)
        self.assertTrue(pm.int_param != pm2.int_two)

        self.assertTrue(pm.int_param < 101)
        self.assertFalse(pm.int_param < pm2.int_one)
        self.assertTrue(pm.int_param < pm2.int_two)

        self.assertTrue(pm.int_param <= 101)
        self.assertTrue(pm.int_param <= 100)
        self.assertFalse(pm.int_param <= pm2.int_minus_one)
        self.assertTrue(pm.int_param <= pm2.int_two)

        self.assertFalse(pm.int_param > 101)
        self.assertFalse(pm.int_param > pm2.int_one)
        self.assertTrue(pm.int_param > pm2.int_minus_one)
        self.assertFalse(pm.int_param > pm2.int_two)

        self.assertFalse(pm.int_param >= 101)
        self.assertTrue(pm.int_param >= 100)
        self.assertTrue(pm.int_param >= pm2.int_minus_one)
        self.assertFalse(pm.int_param >= pm2.int_two)

    def test_operators(self):
        pm = self.pm

        class MyPM2(ParameterManager):
            param_defs = dict(
                int_one=param_def(1, check_int, 'Integer parameter'),
                int_two=param_def(2, check_int, 'Integer parameter'),
            )

        pm2 = MyPM2()

        # Integers

        pm.int_param = 2
        self.assertEqual(pm.int_param, 2)

        pm.int_param = pm.int_param + 100
        self.assertEqual(pm.int_param, 102)

        pm.int_param = pm.int_param + pm2.int_one
        self.assertEqual(pm.int_param, 103)

        pm.int_param = pm.int_param - 52
        self.assertEqual(pm.int_param, 51)

        pm.int_param = pm.int_param - pm2.int_one
        self.assertEqual(pm.int_param, 50)

        pm.int_param = pm.int_param * 4
        self.assertEqual(pm.int_param, 200)

        pm.int_param = pm.int_param * pm2.int_two
        self.assertEqual(pm.int_param, 400)

        pm.int_param = pm.int_param / 5
        self.assertEqual(pm.int_param, 80)

        pm.int_param = pm.int_param / pm2.int_two
        self.assertEqual(pm.int_param, 40)

        pm.int_param = pm.int_param // 5.2
        self.assertEqual(pm.int_param, 7)

        pm.int_param = pm.int_param // pm2.int_two
        self.assertEqual(pm.int_param, 3)

        pm.int_param = 40

        pm.int_param = pm.int_param % 7
        self.assertEqual(pm.int_param, 5)

        pm.int_param = pm.int_param % pm2.int_two
        self.assertEqual(pm.int_param, 1)

        pm.int_param = 5

        pm.int_param = pm.int_param ** 2
        self.assertEqual(pm.int_param, 25)

        pm.int_param = 5

        pm.int_param = pm.int_param ** pm2.int_two
        self.assertEqual(pm.int_param, 25)

        pm.int_param = pm.int_param << pm2.int_two
        self.assertEqual(pm.int_param, 100)

        pm.int_param = pm.int_param >> pm2.int_two
        self.assertEqual(pm.int_param, 25)

        self.assertEqual(pm.int_param & pm2.int_one, 1)

        self.assertEqual(pm.int_param & pm2.int_two, 0)

        self.assertEqual(pm.int_param ^ pm2.int_one, 24)

        self.assertEqual(pm.int_param ^ pm2.int_two, 27)

        self.assertEqual(pm.int_param | pm2.int_one, 25)

        self.assertEqual(pm.int_param | pm2.int_two, 27)

        # Floats

        pm.float_param = 2.5
        self.assertEqual(pm.float_param, 2.5)

        pm.float_param = pm.float_param + 100
        self.assertEqual(pm.float_param, 102.5)

        pm.float_param = pm.float_param - 52
        self.assertEqual(pm.float_param, 50.5)

        pm.float_param = pm.float_param * 4
        self.assertEqual(pm.float_param, 202)

        pm.float_param = pm.float_param / 5
        self.assertEqual(pm.float_param, 40.4)

        # Not sure why I have to use _value here.  Something about
        # the implementation of assertAlmostEqual?
        pm.float_param = pm.float_param ** 2
        self.assertAlmostEqual(pm.float_param._value, 1632.16, 2)

        # Strings

        pm.str_param = 'abc'
        self.assertEqual(pm.str_param, 'abc')

        pm.str_param = pm.str_param + 'def'
        self.assertEqual(pm.str_param, 'abcdef')

        with self.assertRaises(TypeError):
            pm.str_param = pm.str_param - 'def'
        self.assertEqual(pm.str_param, 'abcdef')

        pm.str_param = pm.str_param * 2
        self.assertEqual(pm.str_param, 'abcdefabcdef')

        with self.assertRaises(TypeError):
            pm.str_param = pm.str_param / 2
        self.assertEqual(pm.str_param, 'abcdefabcdef')

        with self.assertRaises(TypeError):
            pm.str_param = pm.str_param ** 2
        self.assertEqual(pm.str_param, 'abcdefabcdef')

    def test_reverse_operators(self):
        pm = self.pm

        # Integers

        pm.int_param = 2
        self.assertEqual(pm.int_param, 2)

        pm.int_param = 100 + pm.int_param
        self.assertEqual(pm.int_param, 102)

        pm.int_param = 52 - pm.int_param
        self.assertEqual(pm.int_param, -50)

        pm.int_param = 4 * pm.int_param
        self.assertEqual(pm.int_param, -200)

        pm.int_param = 2000 / pm.int_param
        self.assertEqual(pm.int_param, -10)

        pm.int_param = 2 ** abs(pm.int_param)
        self.assertEqual(pm.int_param, 1024)

        pm.int_param = 1000000 // pm.int_param
        self.assertEqual(pm.int_param, 976)

        pm.int_param = 1024

        class OtherIntPM(ParameterManager):
            param_defs = dict(
                other_int=param_def(100000, check_int, 'Integer parameter'),
            )

        pm2 = OtherIntPM()

        pm.int_param = 100000 % pm.int_param
        self.assertEqual(pm.int_param, 672)

        pm.int_param = 1024
        pm.int_param = pm.int_param.__rmod__(pm2.other_int)
        self.assertEqual(pm.int_param, 672)

        pm.int_param = 2
        pm.int_param = 25 << pm.int_param
        self.assertEqual(pm.int_param, 100)

        pm.int_param = 2
        pm.int_param = 100 >> pm.int_param
        self.assertEqual(pm.int_param, 25)

        self.assertEqual(1 & pm.int_param, 1)
        self.assertEqual(2 & pm.int_param, 0)

        self.assertEqual(1 ^ pm.int_param, 24)
        self.assertEqual(2 ^ pm.int_param, 27)

        self.assertEqual(1 | pm.int_param, 25)
        self.assertEqual(2 | pm.int_param, 27)

        # Floats

        pm.float_param = 2.5
        self.assertEqual(pm.float_param, 2.5)

        pm.float_param = 100 + pm.float_param
        self.assertEqual(pm.float_param, 102.5)

        pm.float_param = 52 - pm.float_param
        self.assertEqual(pm.float_param, -50.5)

        pm.float_param = 4 * pm.float_param
        self.assertEqual(pm.float_param, -202)

        pm.float_param = 2000 / pm.float_param
        self.assertAlmostEqual(pm.float_param._value, -9.90, 2)

        # Not sure why I have to use _value here.  Something about
        # the implementation of assertAlmostEqual?
        pm.float_param = 2 ** abs(pm.float_param)
        self.assertAlmostEqual(pm.float_param._value, 956.08, 2)

        # Strings

        pm.str_param = 'abc'
        self.assertEqual(pm.str_param, 'abc')

        pm.str_param = 'def' + pm.str_param
        self.assertEqual(pm.str_param, 'defabc')

        with self.assertRaises(TypeError):
            pm.str_param = 'def' - pm.str_param
        self.assertEqual(pm.str_param, 'defabc')

        pm.str_param = 2 * pm.str_param
        self.assertEqual(pm.str_param, 'defabcdefabc')

        with self.assertRaises(TypeError):
            pm.str_param = 2 / pm.str_param
        self.assertEqual(pm.str_param, 'defabcdefabc')

        with self.assertRaises(TypeError):
            pm.str_param = 2 ** pm.str_param
        self.assertEqual(pm.str_param, 'defabcdefabc')

    def test_inplace_operators(self):
        pm = self.pm

        # Integers

        pm.int_param = 2
        self.assertEqual(pm.int_param, 2)

        pm.int_param += 100
        self.assertEqual(pm.int_param, 102)

        pm.int_param -= 52
        self.assertEqual(pm.int_param, 50)

        pm.int_param *= 4
        self.assertEqual(pm.int_param, 200)

        pm.int_param /= 5
        self.assertEqual(pm.int_param, 40)

        pm.int_param = 200
        pm.int_param //= 5.2
        self.assertEqual(pm.int_param, 38)

        pm.int_param = 40
        pm.int_param **= 2
        self.assertEqual(pm.int_param, 1600)

        pm.int_param %= 23
        self.assertEqual(pm.int_param, 13)

        pm.int_param <<= 2
        self.assertEqual(pm.int_param, 52)

        pm.int_param >>= 2
        self.assertEqual(pm.int_param, 13)

        pm.int_param &= 4
        self.assertEqual(pm.int_param, 4)

        pm.int_param ^= 2
        self.assertEqual(pm.int_param, 6)

        pm.int_param |= 3
        self.assertEqual(pm.int_param, 7)

        pm.int_param = -pm.int_param
        self.assertEqual(pm.int_param, -7)

        pm.int_param = +pm.int_param
        self.assertEqual(pm.int_param, -7)

        self.assertEqual(int(pm.int_param), -7)
        self.assertTrue(isinstance(int(pm.int_param), six.integer_types))

        self.assertEqual(float(pm.int_param), -7)
        self.assertTrue(isinstance(float(pm.int_param), float))

        # Floats

        pm.float_param = 2.5
        self.assertEqual(pm.float_param, 2.5)

        pm.float_param += 100
        self.assertEqual(pm.float_param, 102.5)

        pm.float_param -= 52
        self.assertEqual(pm.float_param, 50.5)

        pm.float_param *= 4
        self.assertEqual(pm.float_param, 202)

        pm.float_param /= 5
        self.assertEqual(pm.float_param, 40.4)

        # Not sure why I have to use _value here.  Something about
        # the implementation of assertAlmostEqual?
        pm.float_param **= 2
        self.assertAlmostEqual(pm.float_param._value, 1632.16, 2)

        # Strings

        pm.str_param = 'abc'
        self.assertEqual(pm.str_param, 'abc')

        pm.str_param += 'def'
        self.assertEqual(pm.str_param, 'abcdef')

        with self.assertRaises(TypeError):
            pm.str_param -= 'def'
        self.assertEqual(pm.str_param, 'abcdef')

        pm.str_param *= 2
        self.assertEqual(pm.str_param, 'abcdefabcdef')

        with self.assertRaises(TypeError):
            pm.str_param /= 2
        self.assertEqual(pm.str_param, 'abcdefabcdef')

        with self.assertRaises(TypeError):
            pm.str_param **= 2
        self.assertEqual(pm.str_param, 'abcdefabcdef')

    def test_inplace_operators2(self):
        pm = self.pm

        class MyPM2(ParameterManager):
            param_defs = dict(
                int_one=param_def(1, check_int, 'Integer parameter'),
                float_one=param_def(1, check_float, 'Float parameter'),
            )

        pm2 = MyPM2()

        # Integers

        pm.int_param = 2
        self.assertEqual(pm.int_param, 2)

        pm2.int_one = 100
        pm.int_param += pm2.int_one
        self.assertEqual(pm.int_param, 102)

        pm2.int_one = 52
        pm.int_param -= pm2.int_one
        self.assertEqual(pm.int_param, 50)

        pm2.int_one = 4
        pm.int_param *= pm2.int_one
        self.assertEqual(pm.int_param, 200)

        pm2.int_one = 5
        pm.int_param /= pm2.int_one
        self.assertEqual(pm.int_param, 40)

        pm.int_param = 200
        pm2.float_one = 5.2
        pm.int_param //= pm2.float_one
        self.assertEqual(pm.int_param, 38)

        pm.int_param = 40
        pm2.int_one = 2
        pm.int_param **= pm2.int_one
        self.assertEqual(pm.int_param, 1600)

        pm2.int_one = 23
        pm.int_param %= pm2.int_one
        self.assertEqual(pm.int_param, 13)

        pm2.int_one = 2
        pm.int_param <<= pm2.int_one
        self.assertEqual(pm.int_param, 52)

        pm2.int_one = 2
        pm.int_param >>= pm2.int_one
        self.assertEqual(pm.int_param, 13)

        pm2.int_one = 4
        pm.int_param &= pm2.int_one
        self.assertEqual(pm.int_param, 4)

        pm2.int_one = 2
        pm.int_param ^= pm2.int_one
        self.assertEqual(pm.int_param, 6)

        pm2.int_one = 3
        pm.int_param |= pm2.int_one
        self.assertEqual(pm.int_param, 7)

    def test_set_params(self):
        pm = self.pm

        # Consecutive key/value pairs

        pm.set_params('int_param', 100, 'float_param', 8.98)
        self.assertEqual(pm.params, {'int_param': 100, 'float_param': 8.98,
                                     'str_param': 'some text'})

        with self.assertRaises(TypeError):
            pm.set_params(100, 200)
        self.assertEqual(pm.params, {'int_param': 100, 'float_param': 8.98,
                                     'str_param': 'some text'})

        with self.assertRaises(ValueError):
            pm.set_params('int_param')
        self.assertEqual(pm.params, {'int_param': 100, 'float_param': 8.98,
                                     'str_param': 'some text'})

        # Tuples of key/value pairs

        pm.set_params(('int_param', 200), ('str_param', 'foo bar'))
        self.assertEqual(pm.params, {'int_param': 200, 'float_param': 8.98,
                                     'str_param': 'foo bar'})

        with self.assertRaises(ValueError):
            pm.set_params(('int_param',))
        self.assertEqual(pm.params, {'int_param': 200, 'float_param': 8.98,
                                     'str_param': 'foo bar'})

        with self.assertRaises(ValueError):
            pm.set_params(('int_param', 100, 'int_param'))
        self.assertEqual(pm.params, {'int_param': 200, 'float_param': 8.98,
                                     'str_param': 'foo bar'})

        with self.assertRaises(TypeError):
            pm.set_params((100, 'int_param'))
        self.assertEqual(pm.params, {'int_param': 200, 'float_param': 8.98,
                                     'str_param': 'foo bar'})

        # Dictionaries

        pm.set_params({'str_param': 'new', 'int_param': 1000}) 
        self.assertEqual(pm.params, {'int_param': 1000, 'float_param': 8.98,
                                     'str_param': 'new'})

        with self.assertRaises(TypeError):
            pm.set_params({100: 200})

        # Keyword parameters

        pm.set_params(int_param=99, float_param=9.87)
        self.assertEqual(pm.params, {'int_param': 99, 'float_param': 9.87,
                                     'str_param': 'new'})

        with self.assertRaises(ValueError):
            pm.set_params(int_param='foo')

        # Another ParameterManager

        class MyPM2(ParameterManager):
            param_defs = dict(
                int_param=param_def(10000, check_int, 'Integer parameter'),
                float_param=param_def(65, check_float, 'Float parameter'),
            )

        pm2 = MyPM2()

        pm.set_params(pm2)
        self.assertEqual(pm.params, {'int_param': 10000, 'float_param': 65,
                                     'str_param': 'new'})
        self.assertEqual(pm2.params, {'int_param': 10000, 'float_param': 65})

        class MyPM2(ParameterManager):
            param_defs = dict(
                int_param=param_def(10000, check_int, 'Integer parameter'),
                float_param=param_def(65, check_float, 'Float parameter'),
                different_param=param_def(666, check_float, 'Float parameter'),
            )
 
        pm2 = MyPM2()

        with self.assertRaises(KeyError):
            pm.set_params(pm2)
        self.assertEqual(pm.params, {'int_param': 10000, 'float_param': 65,
                                     'str_param': 'new'})
        self.assertEqual(pm2.params, {'int_param': 10000, 'float_param': 65,
                                      'different_param': 666})

        # ParameterDict 

        class MyPM2(ParameterManager):
            param_defs = dict(
                int_param=param_def(10000, check_int, 'Integer parameter'),
                float_param=param_def(65, check_float, 'Float parameter'),
            )

        pm2 = MyPM2()

        pm2.int_param = 1
        pm2.float_param = 2

        pm.set_params(pm2.params)

        self.assertEqual(pm.params, {'int_param': 1, 'float_param': 2,
                                     'str_param': 'new'})
        self.assertEqual(pm2.params, {'int_param': 1, 'float_param': 2})

        # Parameter

        class MyPM2(ParameterManager):
            param_defs = dict(
                int_param=param_def(10000, check_int, 'Integer parameter'),
                float_param=param_def(65, check_float, 'Float parameter'),
            )

        pm2 = MyPM2()

        pm.set_params(pm2.int_param, pm2.float_param)

        self.assertEqual(pm.params, {'int_param': 10000, 'float_param': 65,
                                     'str_param': 'new'})
        self.assertEqual(pm2.params, {'int_param': 10000, 'float_param': 65})

    def test_has_param(self):
        pm = self.pm

        self.assertTrue(pm.has_param('int_param'))

        self.assertTrue(pm.has_param('float_param'))

        self.assertTrue(pm.has_param('str_param'))

        self.assertFalse(pm.has_param('Str_param'))

        self.assertFalse(pm.has_param(100))

    def test_get_params(self):
        pm = self.pm
 
        self.assertEqual(pm.get_params(), {'int_param': 1, 'float_param': 12.34,
                                           'str_param': 'some text'})

        self.assertEqual(pm.get_params('int_param', 'str_param'),
                         {'int_param': 1, 'str_param': 'some text'})

        with self.assertRaises(KeyError):
            pm.get_params('foo')

    def test_seemingly_unreachable(self):
        pm = self.pm

        class MyPM2(ParameterManager):
            param_defs = dict(
                int_one=param_def(1, check_int, 'Integer parameter'),
                int_two=param_def(2, check_int, 'Integer parameter'),
                float_one=param_def(2.5, check_float, 'Float parameter'),
            )

        pm2 = MyPM2()

        pm.int_param = 100

        self.assertEqual(pm.int_param.__radd__(pm2.int_one), 101)
        self.assertTrue(isinstance(pm.int_param.__radd__(pm2.int_one), Parameter))
        self.assertEqual(pm.int_param.__rsub__(pm2.int_one), -99)
        self.assertTrue(isinstance(pm.int_param.__rsub__(pm2.int_one), Parameter))
        self.assertEqual(pm.int_param.__rmul__(pm2.int_two), 200)
        self.assertTrue(isinstance(pm.int_param.__rmul__(pm2.int_two), Parameter))

        pm.int_param = 2
        pm2.int_two = 20

        self.assertEqual(pm.int_param.__rtruediv__(pm2.int_two), 10)
        self.assertTrue(isinstance(pm.int_param.__rtruediv__(pm2.int_two), Parameter))
        self.assertEqual(pm.int_param.__rfloordiv__(pm2.float_one), 1)
        self.assertTrue(isinstance(pm.int_param.__rfloordiv__(pm2.float_one), Parameter))

        self.assertEqual(pm.int_param.__rpow__(pm2.int_two), 400)
        self.assertTrue(isinstance(pm.int_param.__rpow__(pm2.int_two), Parameter))

        self.assertEqual(pm.int_param.__rlshift__(pm2.int_two), 80)
        self.assertTrue(isinstance(pm.int_param.__rlshift__(pm2.int_two), Parameter))
        self.assertEqual(pm.int_param.__rrshift__(pm2.int_two), 5)
        self.assertTrue(isinstance(pm.int_param.__rrshift__(pm2.int_two), Parameter))

        pm2.int_two = 3 

        self.assertEqual(pm.int_param.__rand__(pm2.int_two), 2)
        self.assertTrue(isinstance(pm.int_param.__rand__(pm2.int_two), Parameter))
        self.assertEqual(pm.int_param.__rxor__(pm2.int_two), 1)
        self.assertTrue(isinstance(pm.int_param.__rxor__(pm2.int_two), Parameter))
        self.assertEqual(pm.int_param.__ror__(pm2.int_two), 3)
        self.assertTrue(isinstance(pm.int_param.__ror__(pm2.int_two), Parameter))

        self.assertEqual(pm.int_param, 2)
        self.assertEqual(pm.int_param.__invert__(), -3)
        self.assertTrue(isinstance(pm.int_param.__invert__(), Parameter))

        self.assertIn(pm2.float_one.__round__(), [2, 3])
        self.assertTrue(isinstance(pm2.float_one.__round__(), Parameter))
        self.assertEqual(pm2.float_one.__ceil__(), 3)
        self.assertTrue(isinstance(pm2.float_one.__ceil__(), Parameter))
        self.assertEqual(pm2.float_one.__floor__(), 2)
        self.assertTrue(isinstance(pm2.float_one.__floor__(), Parameter))
        self.assertEqual(pm2.float_one.__trunc__(), 2)
        self.assertTrue(isinstance(pm2.float_one.__trunc__(), Parameter))

    def test_check_int_list(self):
        pm = self.pm

        class MyPM2(ParameterManager):
            param_defs = dict(
                int_list=param_def(1,
                                   functools.partial(check_int_list, allow_empty=True),
                                   'Integer list parameter'),
                other_list=param_def([10, 20, 30],
                                     functools.partial(check_int_list, maximum=50, minimum=0),
                                     'Other integer list parameter'),
                none_list=param_def(2,
                                    functools.partial(check_int_list, allow_none=True),
                                    'Integer list parameter with None'),
            )

        pm2 = MyPM2()

        self.assertEqual(pm2.int_list, [1])
        self.assertEqual(pm2.other_list, [10, 20, 30])
        self.assertEqual(pm2.none_list, [2])

        pm2.int_list = []
        self.assertEqual(pm2.int_list, [])

        with self.assertRaises(ValueError):
            pm2.other_list = [40, 50, 60]

        with self.assertRaises(ValueError):
            pm2.other_list = [10, -1, 3]

        with self.assertRaises(ValueError):
            pm2.other_list = None

        pm2.none_list = None
        self.assertEqual(pm2.none_list, None)

        with self.assertRaises(ValueError):
            pm2.none_list = []

        with self.assertRaises(ValueError):
            pm2.int_list = [2, 3, 'foo']

    def test_check_float_list(self):
        pm = self.pm

        class MyPM2(ParameterManager):
            param_defs = dict(
                float_list=param_def(1.2,
                                   functools.partial(check_float_list, allow_empty=True),
                                   'Float list parameter'),
                other_list=param_def([10.5, 20.6, 30],
                                     functools.partial(check_float_list, maximum=50, minimum=0),
                                     'Other float list parameter'),
                none_list=param_def(2.4,
                                    functools.partial(check_float_list, allow_none=True),
                                    'Float list parameter with None'),
            )

        pm2 = MyPM2()

        self.assertEqual(pm2.float_list, [1.2])
        self.assertEqual(pm2.other_list, [10.5, 20.6, 30])
        self.assertEqual(pm2.none_list, [2.4])

        pm2.float_list = []
        self.assertEqual(pm2.float_list, [])

        with self.assertRaises(ValueError):
            pm2.other_list = [40.8, 50.9, 60]

        with self.assertRaises(ValueError):
            pm2.other_list = [10, -1.2, 3]

        with self.assertRaises(ValueError):
            pm2.other_list = None

        pm2.none_list = None
        self.assertEqual(pm2.none_list, None)

        with self.assertRaises(ValueError):
            pm2.none_list = []

        with self.assertRaises(ValueError):
            pm2.float_list = [2, 3, 'foo']

    def test_check_number_list(self):
        pm = self.pm

        class MyPM2(ParameterManager):
            param_defs = dict(
                num_list=param_def(1.2,
                                   functools.partial(check_number_list, allow_empty=True),
                                   'Float list parameter'),
                other_list=param_def([10.5, 20.6, 30],
                                     functools.partial(check_number_list, maximum=50, minimum=0),
                                     'Other number list parameter'),
                none_list=param_def(2.4,
                                    functools.partial(check_number_list, allow_none=True),
                                    'Float list parameter with None'),
            )

        pm2 = MyPM2()

        self.assertEqual(pm2.num_list, [1.2])
        self.assertEqual(pm2.other_list, [10.5, 20.6, 30])
        self.assertEqual(pm2.none_list, [2.4])

        pm2.num_list = []
        self.assertEqual(pm2.num_list, [])

        with self.assertRaises(ValueError):
            pm2.other_list = [40.8, 50.9, 60]

        with self.assertRaises(ValueError):
            pm2.other_list = [10, -1.2, 3]

        with self.assertRaises(ValueError):
            pm2.other_list = None

        pm2.none_list = None
        self.assertEqual(pm2.none_list, None)

        with self.assertRaises(ValueError):
            pm2.none_list = []

        with self.assertRaises(ValueError):
            pm2.num_list = [2, 3, 'foo']


if __name__ == '__main__':
    tm.runtests()
