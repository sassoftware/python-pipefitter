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

''' CAS Transformer Utilities '''

from __future__ import print_function, division, absolute_import, unicode_literals

def get_columns(table):
    '''
    Retrieve the column names of the table

    Parameters
    ----------
    table : CASTable
        The table to retrieve the columns from

    Returns
    -------
    (all-columns, character-columns, numeric-columns)

    '''
    dtypes = table.dtypes
    columns = list(dtypes.index)
    char_cols = list(dtypes[dtypes.isin(['char', 'varchar', 'binary', 'varbinary'])].index)
    num_cols = [x for x in columns if x not in set(char_cols)]
    return columns, char_cols, num_cols
