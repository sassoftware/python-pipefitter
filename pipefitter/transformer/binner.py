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
Binner clases

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import six
from ..base import BaseTransformer
from ..utils.params import splat


@six.python_2_unicode_compatible
class Binner(BaseTransformer):
    '''
    Binner

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
    :class:`Binner`
 
    '''

    def __init__(self, method='bucket', n_bins=5, inputs=None):
        BaseTransformer.__init__(self)
        method, n_bins, inputs = self._validate_params(method, n_bins, inputs)
        self.method = method
        self.n_bins = n_bins
        self.inputs = inputs

    def _validate_params(self, method, n_bins, inputs):
        method = method.lower()
        if method not in ['bucket', 'quantile']:
            raise ValueError('method must be either "bucket" or "quantile"')
        n_bins = max(1, int(n_bins))
        inputs = splat(inputs)
        return method, n_bins, inputs

    def __str__(self):
        out = []
        out.append('method=%s' % repr(self.method))
        out.append('n_bins=%d' % self.n_bins)
        if self.inputs is not None:
            out.append('inputs=%s' % repr(self.inputs))
        return '%s(%s)' % (type(self).__name__, ', '.join(out))
        
    def __repr__(self):
        return str(self)
        
    def transform(self, table, method=None, n_bins=None, inputs=None):
        '''
        Perform binning on the input table

        Parameters
        ----------
        table : data set
            The table to bin
        **kwargs : keyword arguments
            Same as for the constructor

        Returns
        -------
        data set

        '''
        if method is None:
            method = self.method
        if n_bins is None:
            n_bins = self.n_bins
        if inputs is None:
            inputs = self.inputs

        method, n_bins, inputs = self._validate_params(method, n_bins, inputs)
            
        return self._get_super(table).transform(table, method=method,
                                                n_bins=n_bins, inputs=inputs)
