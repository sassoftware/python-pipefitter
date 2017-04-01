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
Imputer clases

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import six
from ..base import BaseImputer


@six.python_2_unicode_compatible
class ImputerMethod(object):
    ''' Class for creating imputer method constants '''

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


@six.python_2_unicode_compatible
class Imputer(BaseImputer):
    '''
    Impute missing values in a data set

    The values specified to replace missing values can be statistics or
    constant values.  To specify a statistic, use one of the following 
    pre-defined contants on the Imputer class.

        * Imputer.MAX
        * Imputer.MEAN
        * Imputer.MEDIAN
        * Imputer.MIDRANGE
        * Imputer.MIN
        * Imputer.MODE
        * Imputer.RANDOM

    Parameters
    ----------
    value : ImputerMethod or scalar or dict, optional
        Specifies the value to use in place of missing values.
            * If an ImputerMethod is specified, that method is used for all
              missing values.
            * If a scalar is specified, that value is used to substitute for
              all missings.
            * If a dict is specified, the keys correspond to the columns and
              the values are the substitution values (which may also be
              ImputerMethod instances).

    Examples
    --------
    Sample data set used for imputing examples:

    >>> data.head()
          A     B     C     D     E  F  G  H
    0   1.0   2.0   3.0   4.0   5.0  a  b  c
    1   6.0   NaN   8.0   9.0   NaN  j  e  f
    2  11.0   NaN  13.0  14.0   NaN     h  i
    3  16.0  17.0  18.0   NaN  20.0  j     l
    4   NaN  22.0  23.0  24.0   NaN     n  o

    Impute values using the mean:

    >>> meanimp = Imputer(Imputer.MEAN) 
    >>> newdata = meanimp.transform(data)
    >>> newdata.head()
          A          B     C      D     E  F  G  H
    0   1.0   2.000000   3.0   4.00   5.0  a  b  c
    1   6.0  13.666667   8.0   9.00  12.5  j  e  f
    2  11.0  13.666667  13.0  14.00  12.5     h  i
    3  16.0  17.000000  18.0  12.75  20.0  j     l
    4   8.5  22.000000  23.0  24.00  12.5     n  o

    Impute values using the mode:

    >>> modeimp = Imputer(Imputer.MODE) 
    >>> newdata = modeimp.transform(data)
    >>> newdata.head()
          A     B     C     D     E  F  G  H
    0   1.0   2.0   3.0   4.0   5.0  a  b  c
    1   6.0   2.0   8.0   9.0   5.0  j  e  f
    2  11.0   2.0  13.0  14.0   5.0  j  h  i
    3  16.0  17.0  18.0   4.0  20.0  j  b  l
    4   1.0  22.0  23.0  24.0   5.0  j  n  o

    Impute a constant value:

    >>> cimp = Imputer(100)
    >>> newdata = cimp.transform(data)
    >>> newdata.head()
           A      B     C      D      E  F  G  H
    0    1.0    2.0   3.0    4.0    5.0  a  b  c
    1    6.0  100.0   8.0    9.0  100.0  j  e  f
    2   11.0  100.0  13.0   14.0  100.0     h  i
    3   16.0   17.0  18.0  100.0   20.0  j     l
    4  100.0   22.0  23.0   24.0  100.0     n  o

    Impute values in specified columns:

    >>> dimp = Imputer({'A': 1, 'B': 100,
                        'F': 'none', 'G': 'miss'})
    >>> newdata = cimp.transform(data)
    >>> newdata.head()
          A      B     C     D     E     F     G  H
    0   1.0    2.0   3.0   4.0   5.0     a     b  c
    1   6.0  100.0   8.0   9.0   NaN     j     e  f
    2  11.0  100.0  13.0  14.0   NaN  none     h  i
    3  16.0   17.0  18.0   NaN  20.0     j  miss  l
    4   1.0   22.0  23.0  24.0   NaN  none     n  o

    '''

    #: Constant that indicates the maximum data value of the column
    MAX = ImputerMethod('max')

    #: Constant that indicates the mean data value of the column
    MEAN = ImputerMethod('mean')

    #: Constant that indicates the median data value of the column
    MEDIAN = ImputerMethod('median')

    #: Constant that indicates the midrange data value of the column
    MIDRANGE = ImputerMethod('midrange')

    #: Constant that indicates the minimum data value of the column
    MIN = ImputerMethod('min')

    #: Constant that indicates the mode data value of the column
    MODE = ImputerMethod('mode')

    #: Constant that indicates that random data should be used
    RANDOM = ImputerMethod('random')

    def __init__(self, value=MEAN):
        BaseImputer.__init__(self)
        self.value = value

    def __str__(self):
        if isinstance(self.value, ImputerMethod):
            value = repr(self.value).upper()
        else:
            value = repr(self.value)
        return '%s(%s)' % (type(self).__name__, value)
        
    def __repr__(self):
        return str(self)
        
    def transform(self, table, value=None):
        '''
        Perform the imputation on the given data set

        Parameters
        ----------
        table : data set
            The data set to impute
        value : ImputerMethod or scalar or dict, optional
            Same as for constructor

        Examples
        --------
        Sample data set used for imputing examples:

        >>> data.head()
              A     B     C     D     E  F  G  H
        0   1.0   2.0   3.0   4.0   5.0  a  b  c
        1   6.0   NaN   8.0   9.0   NaN  j  e  f
        2  11.0   NaN  13.0  14.0   NaN     h  i
        3  16.0  17.0  18.0   NaN  20.0  j     l
        4   NaN  22.0  23.0  24.0   NaN     n  o

        Impute values using the mean:

        >>> meanimp = Imputer(Imputer.MEAN)
        >>> newdata = meanimp.transform(data)
        >>> newdata.head()
              A          B     C      D     E  F  G  H
        0   1.0   2.000000   3.0   4.00   5.0  a  b  c
        1   6.0  13.666667   8.0   9.00  12.5  j  e  f
        2  11.0  13.666667  13.0  14.00  12.5     h  i
        3  16.0  17.000000  18.0  12.75  20.0  j     l
        4   8.5  22.000000  23.0  24.00  12.5     n  o

        Impute values using the mode:

        >>> modeimp = Imputer(Imputer.MODE)
        >>> newdata = modeimp.transform(data)
        >>> newdata.head()
              A     B     C     D     E  F  G  H
        0   1.0   2.0   3.0   4.0   5.0  a  b  c
        1   6.0   2.0   8.0   9.0   5.0  j  e  f
        2  11.0   2.0  13.0  14.0   5.0  j  h  i
        3  16.0  17.0  18.0   4.0  20.0  j  b  l
        4   1.0  22.0  23.0  24.0   5.0  j  n  o

        Impute a constant value:

        >>> cimp = Imputer(100)
        >>> newdata = cimp.transform(data)
        >>> newdata.head()
               A      B     C      D      E  F  G  H
        0    1.0    2.0   3.0    4.0    5.0  a  b  c
        1    6.0  100.0   8.0    9.0  100.0  j  e  f
        2   11.0  100.0  13.0   14.0  100.0     h  i
        3   16.0   17.0  18.0  100.0   20.0  j     l
        4  100.0   22.0  23.0   24.0  100.0     n  o

        Impute values in specified columns:

        >>> dimp = Imputer({'A': 1, 'B': 100,
        ...                 'F': 'none', 'G': 'miss'})
        >>> newdata = cimp.transform(data)
        >>> newdata.head()
              A      B     C     D     E     F     G  H
        0   1.0    2.0   3.0   4.0   5.0     a     b  c
        1   6.0  100.0   8.0   9.0   NaN     j     e  f
        2  11.0  100.0  13.0  14.0   NaN  none     h  i
        3  16.0   17.0  18.0   NaN  20.0     j  miss  l
        4   1.0   22.0  23.0  24.0   NaN  none     n  o

        Returns
        -------
        data set
            Data set of the same type as ``table``

        '''
        if value is None:
            value = self.value
            
        return self._get_super(table).transform(table, value=value)
