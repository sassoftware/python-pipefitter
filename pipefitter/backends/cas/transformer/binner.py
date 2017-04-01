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

''' CAS Binner Implementation '''

from __future__ import print_function, division, absolute_import, unicode_literals

import pandas as pd
import six
import uuid
from .utils import get_columns
from .... import transformer


class Binner(transformer.Binner):
    ''' CAS Binner Implementation '''
    
    def transform(self, table, method='bucket', n_bins=5, inputs=None):
        '''
        Perform binning on the inputs table

        Parameters
        ----------
        method : string, optional
            The type of binning to do
        n_bins : int, optional
            The number of bins
        inputs : string or list-of-strings, optional
            The columns to include.  Defaults to all numeric columns.

        Returns
        -------
        :class:`CASTable`

        '''
        params = {
            'casout': dict(name='bin_%s' % str(uuid.uuid4()).replace('-', '_'),
                           replace=True),
            'outvarsnameprefix': '',
            'outvarsnamesuffix': '',
            '_apptag': 'UI',
            '_messagelevel': 'error',
            'method': method,
            'nbinsarray': n_bins,
        }

        cols, char_cols, num_cols = get_columns(table)

        if inputs is not None:
            params['inputs'] = inputs 
        else:
            params['inputs'] = num_cols

        params['copyvars'] = set(cols).difference(set(params['inputs']))

        table = table.datapreprocess.binning(**params)['OutputCasTables'].ix[0, 'casTable']

        # Reset original column order
        table.set_param(vars=list(cols))

        return table.partition(casout=params['casout'])['casTable']
