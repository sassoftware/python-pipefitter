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
CAS Class Implementations

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import numbers
import warnings
from ... import base


class ResourceManager(base.ResourceManager):
    ''' Resource manager for CAS tables and connections '''

    def is_parallelizable(self, *tables):
        ''' Can usage of these tables be parallelized? '''
        for table in tables:
            res = table.retrieve('table.tableinfo', _messagelevel='error', _apptag='UI')
            if not res['TableInfo'].ix[0, 'Global']:
                return False
        return True

    def emancipate(self, *tables):
        ''' Create a new session for the given tables '''
        out = []
        if not tables:
            return out
        conn = tables[0].get_connection().copy()
        self.track_connection(conn)
        for table in tables:
            table = table.copy()
            table.set_connection(conn)
            out.append(table)
        return out

    def split_data(self, data, k=3, var=None):
        ''' Split the table into sub-tables '''
        outtables = []
        promote = False

        res = data.retrieve('table.tableinfo', _messagelevel='error', _apptag='UI')
        if res['TableInfo'].ix[0, 'Global']:
            promote = True

        if isinstance(k, numbers.Integral):
            out = data.datastep([
                '__my_rank = _rankid_ * 1000 + _threadid_',
                'call streaminit(__my_rank)',
                '__fold_id = floor(rand("uniform") * %s)' % k,
                'drop __my_rank'
            ])

            self.track_table(out)

            if promote:
                out.retrieve('table.promote', _messagelevel='error', _apptag='UI')

            for i in range(k):
                tbla = out.query('__fold_id ~= %s' % i)
                tblb = out.query('__fold_id = %s' % i)
                outtables.append((tbla, tblb))

        elif isinstance(k, numbers.Real):
            out = data.datastep([
                '__my_rank = _rankid_ * 1000 + _threadid_',
                'call streaminit(myrank)',
                '__fold_id = (rand("uniform") < %s) + 1' % k,
                'drop __my_rank'
            ])

            self.track_table(out)

            if promote:
                out.retrieve('table.promote', _messagelevel='error', _apptag='UI')

            outtables.append((out.query('__fold_id = 1'),
                              out.query('__fold_id ~= 1')))

        return outtables

    def unload_model(self, model):
        ''' Remove a model table from memory, if it exists '''
        model.unload() 

    def unload_data(self, data):
        ''' Remove the table from memory, if it exists '''
        if data.retrieve('table.tableexists', _messagelevel='error',
                                              _apptag='UI').get('exists', None):
            data.retrieve('table.droptable', _messagelevel='error',
                                             _apptag='UI')

    def terminate_connection(self, conn):
        ''' End the session and close the connection '''
        conn.retrieve('session.endsession', _messagelevel='error', _apptag='UI')
        conn.close() 
