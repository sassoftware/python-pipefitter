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
Tests for base functionality

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import re
import six
import swat.utils.testing as tm
import unittest
import pipefitter
import pipefitter.backends.cas
import pipefitter.backends.sas
from pipefitter.base import (_BACKEND_MAP, register_backend, unregister_backend,
                             get_super_module, PolySuperMixIn, ParameterManager,
                             BaseTransformer, BaseImputer, BaseEstimator,
                             BaseGridSearchCV, BaseModel, ResourceManager)


class TestRegistry(tm.TestCase):

    def setUp(self):
        self.defaults = _BACKEND_MAP.copy() 

    def tearDown(self):
        _BACKEND_MAP.clear()
        _BACKEND_MAP.update(self.defaults)

    def test_register(self):
        self.assertTrue('FooBar' not in _BACKEND_MAP)
        register_backend('FooBar', '.backends.foobar')
        self.assertTrue('FooBar' in _BACKEND_MAP)

    def test_unregister(self):
        self.assertTrue('CASTable$' in _BACKEND_MAP)
        unregister_backend('CASTable$')
        self.assertTrue('CASTable$' not in _BACKEND_MAP)


class TestSuperModule(tm.TestCase):

    def test_get_super_module(self):
        # Create mock objects
        class CASTable(object):
            pass
        class SASdata(object):
            pass
        class FooBar(object):
            pass

        self.assertEqual(get_super_module(CASTable()), pipefitter.backends.cas)
        self.assertEqual(get_super_module(SASdata()), pipefitter.backends.sas)

        with self.assertRaises(ValueError):
            get_super_module(FooBar())

    def test_mixin(self):
        # Create mock objects
        class CASTable(object):
            pass
        class SASdata(object):
            pass

        class Imputer(ParameterManager, PolySuperMixIn):
            pass

        imp = Imputer()

        # Test backends
        casimp = imp._get_super(CASTable())
        self.assertEqual(casimp.__class__.__name__, 'Imputer')
        self.assertEqual(casimp.__module__, 'pipefitter.backends.cas.transformer.imputer')

        sasimp = imp._get_super(SASdata())
        self.assertEqual(sasimp.__class__.__name__, 'Imputer')
        self.assertEqual(sasimp.__module__, 'pipefitter.backends.sas.transformer.imputer')

        # Test explicit name
        tree = imp._get_super(CASTable(), name='DecisionTree')
        self.assertEqual(tree.__class__.__name__, 'DecisionTree')
        self.assertEqual(tree.__module__, 'pipefitter.backends.cas.estimator.tree')

        # Test bad explicit name
        with self.assertRaises(AttributeError): 
            imp._get_super(CASTable(), name='Foo')

        # Test caching
        casimp = imp._get_super(CASTable(), cache=True)
        self.assertTrue(hasattr(imp, '@super'))
        self.assertTrue(getattr(imp, '@super') is casimp)
        self.assertTrue(imp._get_super(CASTable()) is casimp)


class TestBaseClasses(tm.TestCase):

    def test_BaseTransformer(self):
        with self.assertRaises(NotImplementedError):
            BaseTransformer().transform('foo')

        with self.assertRaises(AttributeError):
            BaseTransformer().fit

    def test_BaseImputer(self):
        with self.assertRaises(NotImplementedError):
            BaseImputer().transform('foo')

        with self.assertRaises(AttributeError):
            BaseImputer().fit

    def test_BaseGridSearchCV(self):
        with self.assertRaises(NotImplementedError):
            BaseGridSearchCV().fit('foo')

        with self.assertRaises(NotImplementedError):
            BaseGridSearchCV().score('foo')

        with self.assertRaises(AttributeError):
            BaseGridSearchCV().transform

    def test_BaseEstimator(self):
        with self.assertRaises(NotImplementedError):
            BaseEstimator().fit('foo')

        with self.assertRaises(NotImplementedError):
            BaseEstimator().transform('foo')

        with self.assertRaises(AttributeError):
            BaseEstimator().score

    def test_BaseModel(self):
        class CASTable(object):
            pass

        table = CASTable()
        params = dict()

        model = BaseModel(table, params, dict(c=3),
                          backend=pipefitter.backends.cas)

        with self.assertRaises(AttributeError):
            model.fit

        with self.assertRaises(NotImplementedError):
            model.score(table)

        # Test arguments
        self.assertTrue(model.data is table)
        self.assertEqual(model.diagnostics, dict(c=3))
        self.assertTrue(model.backend is pipefitter.backends.cas)

        # Test transform
        self.assertTrue(model.transform(table) is table)
        
        # Test unload
        self.assertTrue(model.unload() is None)

        # Test check empty backend
        model = BaseModel(table, params, dict(c=3))
        self.assertTrue(model.backend is None)
        self.assertTrue(model._check_backend(table) is None)
        
        # Test backend for bad data
        class UnknownData(object):
            pass

        model = BaseModel(table, params, dict(c=3),
                          backend=pipefitter.backends.cas)

        with self.assertRaises(ValueError):
            model._check_backend(UnknownData())


class TestResourceManager(tm.TestCase):

    def test_context_manager(self):
        class CASTable(object):
            pass
        class TestModel(BaseModel):
            pass
        class TestConnection(object):
            pass

        table1 = CASTable()
        table2 = CASTable()
        model = TestModel(table1, dict())
        conn1 = TestConnection()
        conn2 = TestConnection()

        with ResourceManager() as mgr:

            self.assertEqual(mgr.tables, [])
            self.assertEqual(mgr.models, [])
            self.assertEqual(mgr.connections, [])

            mgr.track_table(table1)
            mgr.track_table(table2)
            mgr.track_model(model)
            mgr.track_connection(conn1)
            mgr.track_connection(conn2)

            self.assertEqual(mgr.tables, [table1, table2])
            self.assertEqual(mgr.models, [model])
            self.assertEqual(mgr.connections, [conn1, conn2])

        self.assertEqual(mgr.tables, [])
        self.assertEqual(mgr.models, [])
        self.assertEqual(mgr.connections, [])

    def test_emancipate(self):
        class CASTable(object):
            pass

        table1 = CASTable()
        table2 = CASTable()

        with ResourceManager() as mgr:
            out = mgr.emancipate(table1, table2) 
            self.assertEqual(out, [table1, table2])

    def test_parallelizable(self):
        class CASTable(object):
            pass

        table1 = CASTable()
        table2 = CASTable()

        with ResourceManager() as mgr:
            out = mgr.is_parallelizable(table1, table2)    
            self.assertTrue(out is False)

    def test_split_data(self):
        with self.assertRaises(NotImplementedError):
            with ResourceManager() as mgr:
                mgr.split_data('foo')

    def test_unload_data(self):
        with ResourceManager() as mgr:
            self.assertTrue(mgr.unload_data('foo') is None)

    def test_unload_model(self):
        with ResourceManager() as mgr:
            self.assertTrue(mgr.unload_model('foo') is None)

    def test_terminate_connection(self):
        with ResourceManager() as mgr:
            self.assertTrue(mgr.terminate_connection('foo') is None)


if __name__ == '__main__':
    tm.runtests()
