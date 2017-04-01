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

''' Base Classes '''

from __future__ import print_function, division, absolute_import, unicode_literals

import copy
import os
import re
import six
from .utils.params import ParameterManager, READ_ONLY_PARAMETER

_BACKEND_MAP = {
    'CASTable$': '.backends.cas',
    'SASdata$' : '.backends.sas',
}


def register_backend(clsname, package):
    '''
    Register a new backend

    Parameters
    ----------
    clsname : string
        The name of the class to associate with the backend
    package : string
        The Python package that contains the implementations.
        This can be a package path relative to `pipeline`, or
        an absolute package path.

    '''
    _BACKEND_MAP[clsname] = package   


def unregister_backend(clsname):
    '''
    Unregister a backend

    Parameters
    ----------
    clsname : string
        The name of the class to unregister

    '''
    del _BACKEND_MAP[clsname]


def get_super_module(obj):
    '''
    Return the backend module for the given object

    Parameters
    ----------
    obj : data set
        The data set object used to detect the backend type

    Returns
    -------
    Python module

    '''
    modulename = None
    objname = '%s.%s' % (type(obj).__module__, type(obj).__name__)
    for key, module in sorted(_BACKEND_MAP.items()):
        if re.search(key, objname):
            modulename = module
            break

    if not modulename:
        raise ValueError('Object type %s is not registered to a backend' % objname)

    import importlib

    return importlib.import_module(modulename, package='pipefitter')


class PolySuperMixIn(object):
    ''' Mixin for retrieving the appropriate backend for a data set type '''

    def _get_super(self, obj, name=None, cache=False):
        '''
        Get the appropriate superclass for the given object

        Parameters
        ----------
        obj : data set
            The data object used to detect the appropriate backend
        name : string, optional
            The name of the class to return.  This defaults to the 
            class name of `self`.
        cache : boolean, optional
            Should the result of this operation be cached for 
            future calls? 

        Returns
        -------
        Specified class from the appropriate backend package

        '''
        if getattr(self, '@super', None) is not None:
            return getattr(self, '@super')

        if name is None:
            name = self.__class__.__name__

        try:
            out = getattr(get_super_module(obj), name)(**self.get_params())
        except AttributeError:
            raise AttributeError('The backend associated with %s.%s does not support %s' %
                                 (type(obj).__module__, type(obj).__name__, name))

        if cache:
            setattr(self, '@super', out)

        return out

    def _get_backend(self, data):
        return get_super_module(data)


class BaseTransformer(ParameterManager, PolySuperMixIn):
    ''' Base class for transformer '''

    def transform(self, table, *args, **kwargs):
        ''' Transform function for transformer '''
        raise NotImplementedError


class BaseImputer(BaseTransformer, PolySuperMixIn):
    ''' Base class for imputer '''
    pass


class BaseEstimator(BaseTransformer):
    ''' Base class for estimators '''

    def fit(self, table, *args, **kwargs):
        ''' Fit function for estimator '''
        raise NotImplementedError


class BaseGridSearchCV(ParameterManager, PolySuperMixIn):
    ''' Base class for grid search '''

    def fit(self, table, *args, **kwargs):
        raise NotImplementedError

    def score(self, table, *args, **kwargs):
        raise NotImplementedError


class BaseModel(ParameterManager, PolySuperMixIn):
    ''' Base class for trained model '''

    def __init__(self, data, params, diagnostics=None, backend=None):
        ParameterManager.__init__(self, **params)
        self._set_options(READ_ONLY_PARAMETER)
        self.data = data
        self.diagnostics = diagnostics
        self.backend = backend

    def _check_backend(self, data):
        if self.backend is None:
            return
        if self.backend is not get_super_module(data):
            raise TypeError('Data type of data set does not match the data type '
                            'used to create the model.')

    def score(self, table, *args, **kwargs):
        ''' Score the data using the model '''
        raise NotImplementedError

    def transform(self, table, *args, **kwargs):
        '''
        Transform function for the model

        This method is primarily used as a pass-through for data
        sets when multiple estimators are used in a pipeline.

        Parameters
        ----------
        table : data set
            The data set to transform

        Returns
        -------
        data set

        '''
        return table

    def unload(self):
        ''' Unload any resources used by the model '''
        return


class ResourceManager(object):
    ''' Resource manager for pipelines and model selection schemes '''

    def __enter__(self):
        ''' Begin a new resource context '''
        self.tables = []
        self.models = []
        self.connections = []
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        ''' Clean up all resources '''
        while self.tables:
            self.unload_data(self.tables.pop())
        while self.models:
            self.unload_model(self.models.pop())
        while self.connections:
            self.terminate_connection(self.connections.pop())

    def track_model(self, *models):
        ''' Track a model object '''
        for model in models:
            self.models.append(model)

    def track_table(self, *tables):
        ''' Track a table object ''' 
        for table in tables:
            self.tables.append(table)

    def track_connection(self, *conns):
        ''' Track a connection object '''
        for conn in conns:
            self.connections.append(conn)

    def emancipate(self, *tables):
        '''
        Create copies of the tables that can be used in parallel

        Parameters
        ----------
        *tables : one or more data set
            The data sets to process

        Returns
        -------
        list 
            The first item of the list is a session object or None.
            This is the session or connection that the new data sets
            belong to.  The remaining items in the list are the 
            emancipated data set objects.

        '''
        return list(tables)

    def is_parallelizable(self, *tables):
        '''
        Can the given tables be used in parallel?

        Parameters
        ----------
        *tables : one or more data set
            The data sets to process

        Returns
        -------
        boolean

        '''
        return False

    def split_data(self, data, k=3, var=None):
        '''
        Split the table into sub-tables

        Parameters
        ----------
        data : data set
            The data set to split
        k : int or float, optional
            Determines the splitting strategy.
                * int - the table is split into `k` equal pieces
                * float - the table is split in two sub-tables.
                  The first sub-table contains the lower `k`-percent
                  of rows.  The second sub-table contains the
                  remaining rows.
        var : string, optional
            Variable to stratify kfold by
        parallel : boolean, optional
            If True, the tables should be returned in separate
            sessions so that they can be used in parallel.

        Returns
        -------
        list of tuples
            If the tables can be run in separate sessions (triggered by
            the parallel=True option), each tuple will contain two
            data set objects and a connection object.
            If tables must be used in the same session (either because
            parallel=False or the tables have local scope), the tuples
            contain just the two data set objects.

        '''
        raise NotImplementedError('Data splitting is not supported')

    def unload_data(self, data):
        '''
        Remove the table from memory

        Parameters
        ----------
        data : data set
            The data set to remove from the server's memory

        '''
        return

    def unload_model(self, model):
        '''
        Remove the model from memory

        Parameters
        ----------
        model : model object
            The model to free the resources of

        '''
        return

    def terminate_connection(self, conn):
        '''
        End the session and close the connection

        Parameters
        ----------
        conn : connection object
            The connection to close

        '''
        return
