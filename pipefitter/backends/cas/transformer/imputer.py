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

''' CAS Imputer Implementations '''

from __future__ import print_function, division, absolute_import, unicode_literals

import pandas as pd
import six
import uuid
from .utils import get_columns
from .... import transformer


class Imputer(transformer.Imputer):
    ''' CAS Imputer Implementation '''
    
    def transform(self, table, value=transformer.Imputer.MEAN):
        '''
        Fill data missing values with specified values

        Parameters
        ----------
        table : CASTable
            The table to impute
        value : ImputerMethod or scalar or dict or Series or DataFrame, optional
            Specifies the value to use in place of missing values.
                * If an ImputerMethod is specified, that method is used for all
                  missing values.
                * If a scalar is specified, that value is used to substitute for
                  all missings.
                * If a dict is specified, the keys correspond to the columns and
                  the values are the substitution values (which may also be
                  ImputerMethod instances).
                * If a Series is specified, the index corresponds to the columns
                  and the values are the substitution values.

        Returns
        -------
        :class:`CASTable`

        '''
        if isinstance(value, (pd.DataFrame, type(table))):
            raise TypeError('DataFrame-like replacements are not supported')

        params = {
            'casout': dict(name='impute_%s' % str(uuid.uuid4()).replace('-', '_'),
                           replace=True),
            'outvarsnameprefix': '',
            'outvarsnamesuffix': '',
            '_apptag': 'UI',
            '_messagelevel': 'error',
        }

        # Retrieve column information
        cols, char_cols, num_cols = get_columns(table)

        # Get replacement categories
        num_stats, char_stats, num_const, char_const = \
            self._get_value_categories(value, char_cols, num_cols, table)

        # Replace the missing values
        table = self._impute_num_constants(table, num_const, cols, params)
        table = self._impute_char_constants(table, char_const, cols, params)
        for stat, stat_cols in num_stats.items():
            table = self._impute_num_stats(table, stat, stat_cols, cols, params)
        for stat, stat_cols in char_stats.items():
            table = self._impute_char_stats(table, stat, stat_cols, cols, params)

        # Reset original column order
        table.set_param(vars=list(cols))

        return table.partition(casout=params['casout'])['casTable']

    def _get_value_categories(self, value, char_cols, num_cols, table):
        '''
        Categorize imputer method types

        Parameters
        ----------
        value : scalar or dict or pd.Series
            The values to use for missing values
        char_cols : list
            The character columns in the table
        num_cols : list
            The numeric columns in the table
        table : CASTable
            The table to impute

        Returns
        -------
        (num_stats dict, char_stats dict, num_const dict, char_const dict)
            * num_stats / char_stats specify the statistic as the key and
              the values are the columns to apply it to.
            * num_const / char_const specify the value to use as the key and
              the values are the columns to apply them to.

        '''
        # Convert Series / etc. to a dictionary
        if hasattr(value, 'to_dict'):
            value = value.to_dict()

        # Make sure we are always starting with a dictionary. 
        # A key value of `None` means all columns.
        if not isinstance(value, dict):
            value = {None: value} 

        num_stats = {}
        char_stats = {}
        char_const = {}
        num_const = {}

        def get_col_names(colname, all_cols):
            if colname is None:
                return all_cols
            elif colname.lower() not in all_cols:
                return set()
            return set([colname])

        char_cols_low = [x.lower() for x in char_cols]
        num_cols_low = [x.lower() for x in num_cols]

        for col, repl in value.items():
            if isinstance(repl, transformer.ImputerMethod):
                repl = str(repl)

                if repl in ['mode']:
                    char_stats.setdefault(repl, set())\
                              .update(get_col_names(col, char_cols_low + num_cols_low))

                elif repl in ['mean', 'median', 'midrange', 'random']:
                    num_stats.setdefault(repl, set()).update(get_col_names(col, num_cols_low))

                elif repl in ['max', 'min']:
                    num_stats.setdefault(repl, set()).update(get_col_names(col, num_cols_low))
                    if char_cols:
                        for key, value in getattr(table[char_cols], repl)().iteritems():
                            char_const.setdefault(value, set()).update([key])

            elif isinstance(repl, six.string_types):
                if col is not None and col.lower() not in char_cols_low:
                    raise TypeError("Column '%s' is numeric, but substitution is character."
                                    % col)
                char_const.setdefault(repl, set()).update(get_col_names(col, char_cols_low))

            else:
                if col is not None and col.lower() not in num_cols_low:
                    raise TypeError("Column '%s' is character, but substitution is numeric."
                                    % col)
                num_const.setdefault(repl, set()).update(get_col_names(col, num_cols_low))

        return num_stats, char_stats, num_const, char_const

    def _impute_constants(self, table, itype, value, all_cols, params):
        '''
        Impute constant values

        Parameters
        ----------
        table : CASTable
            The table to impute
        itype : string
            The type of imputation: 'nominal' or 'continuous'
        value : dict
            The dictionary of values to impute.  The keys are the values to
            use in place of missing values.  The values are sets of colummn
            names where the value should be used.
        all_cols : list
            A list of all columns in the table
        params : dict
            Base parameters for the datapreprocess.impute action call

        Returns
        -------
        :class:`CASTable`

        '''
        if not value:
            return table
        params = dict(params)
        params['method' + itype] = 'value'
        params['inputs'] = inputs = []
        params['values' + itype] = values = []
        for val, cols in value.items():
            inputs.extend(cols)
            if len(value) > 1:
                values.extend([val] * len(cols))
            else:
                values.append(val)
        params['copyvars'] = set([x.lower() for x in all_cols])\
                                .difference([x.lower() for x in set(params['inputs'])])
        if not params['copyvars']:
            del params['copyvars']
        return self._call_impute_action(table, params)

    def _impute_char_constants(self, table, value, all_cols, params):
        '''
        Impute character constants

        See Also
        --------
        :meth:`_impute_constants`

        Returns
        -------
        :class:`CASTable`

        '''
        return self._impute_constants(table, 'nominal', value, all_cols, params)

    def _impute_num_constants(self, table, value, all_cols, params):
        '''
        Impute numeric constants

        See Also
        --------
        :meth:`_impute_constants`

        Returns
        -------
        :class:`CASTable`

        '''
        return self._impute_constants(table, 'continuous', value, all_cols, params)

    def _impute_stats(self, table, stype, stat, inputs, all_cols, params):
        '''
        Impute statisticals values

        Parameters
        ----------
        table : CASTable
            The table to impute
        stype : string
            The statistic to use for missing values
        inputs : list
            The list of columns to impute
        all_cols : list
            List of all columns in the table
        params : dict
            Base action parameters for the datapreprocess.impute action

        Returns
        -------
        :class:`CASTable`

        '''
        if not inputs:
            return table
        params = dict(params)
        params['method' + stype] = stat
        params['inputs'] = list(inputs)
        params['copyvars'] = set([x.lower() for x in all_cols])\
                                .difference([x.lower() for x in set(params['inputs'])])
        if not params['copyvars']:
            del params['copyvars']
        return self._call_impute_action(table, params)

    def _impute_num_stats(self, table, stat, num_cols, all_cols, params):
        '''
        Impute numeric statistics

        See Also
        --------
        :meth:`_impute_stats`

        Returns
        -------
        :class:`CASTable`

        '''
        return self._impute_stats(table, 'continuous', stat, num_cols, all_cols, params)

    def _impute_char_stats(self, table, stat, char_cols, all_cols, params):
        '''
        Impute numeric statistics

        See Also
        --------
        :meth:`_impute_stats`

        Returns
        -------
        :class:`CASTable`

        '''
        if stat == 'mode':
            #estimation of maximum cardinality for setting high cardinality
            #highcardinality's cost is the same as summary's:- the fastest cardinality
            #estimator out there.
            res = table.dataPreprocess.highCardinality(inputs=char_cols)['HighCardinalityDetails']
            max_card = max(res['CardinalityEstimate'])
            params['distinctCountLimit'] = max(5000, 1.5*max_card)
        return self._impute_stats(table, 'nominal', stat, char_cols, all_cols, params)

    def _call_impute_action(self, table, params):
        '''
        Call the CAS action on the server

        Parameters
        ----------
        table : CASTable
            The table to impute
        params : dict
            The action parameters

        Raises
        ------
        RuntimeError
            If an error occurred when running the datapreprocess.impute action

        Returns
        -------
        :class:`CASTable`

        '''
        out = table.datapreprocess.impute(**params)
        if out.status:
            raise RuntimeError(out.status)
        return out['OutputCasTables']['casTable'][0]
