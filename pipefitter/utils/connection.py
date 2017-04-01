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
Connection management utilities

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import weakref


class ConnectionManager(object):
    '''
    Register a connection with an object

    '''

    def __init__(self):
        self._connection = None

    def set_connection(self, connection):
        ''' Set the connection object for estimator '''
        if connection is None:
            self._connection = None
        else:
            self._connection = weakref.ref(connection)

    def get_connection(self):
        '''
        Get the connection session for estimator

        Returns
        -------
        Connection object

        '''
        conn = None
        if self._connection is not None:
            try:
                conn = self._connection()
            except:
                pass
        if conn is None:
            raise ValueError('No connection is currently registered')
        return conn
