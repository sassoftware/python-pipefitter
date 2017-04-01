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
Neural Networks

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import functools
from ..base import BaseEstimator, BaseModel
from ..utils.params import (param_def, check_int, check_string, check_boolean,
                            check_variable, check_variable_list, check_float,
                            check_int_list, extract_params)


class NeuralNetwork(BaseEstimator):
    '''
    Neural Network

    Parameters
    ----------
    acts : string, optional
        Specifies the activation function for the neurons on each
        hidden layer. Valid values are 'identity', 'logistic', 'sin',
        'softplus', and 'tanh'.
    annealing_rate : float, optional
        Specifies the annealing parameter
    direct : bool, optional
        Specifies to use an architecture that is an extension of MLP
        with direct connections between the input layer and the output layer
    error_func : string, optional
        Specifies the error function to train the network.  Valid
        values are 'normal' and 'entropy'.
    hiddens : list-of-ints, optional
        Specifies the number of hidden neurons for each hidden layer
        in the feedforward model
    lasso : float, optional
        Specifies the L2 regularization parameter, the value must be
        nonnegative
    learning_rate : float, optional
        Specifies the learning rate parameter for SGD
    max_iters : int, optional
        Specifies the maximum iterations allowed for optimization
    max_time : int, optional
        Specifies the maximum time (in seconds) allowed for
        optimization
    ridge : float, optional
        Specifies the L2 regularization parameter, the value must be
        nonnegative.
    seed : float, optional
        Specifies the random number seed for generating random numbers to
        initialize the network weights
    std : string, optional
        Specifies the standardization to use on the interval variables.
        Valid values are 'midrange', 'none', and 'std'.
    optimization : string, optional
        Specifies the optimization technique. Valid values are 'lbfgs'
        and 'sgd'.
    target : string, optional
        The target variable
    nominals : string or list of strings, optional
        The nominal variables
    inputs : string or list of strings, optional
        The input variables

    Examples
    --------
    >>> nn = NeuralNetwork(target='Origin',
    ...                    inputs=['MPG_City', 'MPG_Highway', 'Length',
    ...                            'Weight', 'Type', 'Cylinders'],
    ...                    nominals = ['Type', 'Cylinders', 'Origin'])

    Returns
    -------
    :class:`NeuralNetwork`

    '''

    param_defs = dict(
        acts=param_def('tanh', functools.partial(check_string, normalize=True,
                                                 valid_values=['identity', 'logistic',
                                                               'sin', 'softplus', 'tanh'])),
        annealing_rate=param_def(1e-06, functools.partial(check_float, minimum=0)),
        direct=param_def(False, check_boolean),
        error_func=param_def(None, functools.partial(check_string,
                                       valid_values=['normal', 'entropy'],
                                       allow_none=True, normalize=True)),
        hiddens=param_def(9, functools.partial(check_int_list, allow_none=True,
                                                  minimum=1)),
        lasso=param_def(0, functools.partial(check_float, minimum=0)),
        learning_rate=param_def(0.001, functools.partial(check_float, minimum=0)),
        max_iters=param_def(10, functools.partial(check_int, minimum=0)),
        max_time=param_def(0, functools.partial(check_float, minimum=0)),
        num_tries=param_def(10, functools.partial(check_int, minimum=0)),
        ridge=param_def(0, functools.partial(check_float, minimum=0)),
        seed=param_def(0.0, check_float),
        std=param_def('midrange', functools.partial(check_string, normalize=True,
                                                    valid_values=['midrange', 'none',
                                                                  'std'])),
        optimization=param_def('lbfgs', functools.partial(check_string, normalize=True,
                                                          valid_values=['lbfgs', 'sgd'])),
        target=param_def(None, check_variable),
        nominals=param_def(None, check_variable_list),
        inputs=param_def(None, check_variable_list),
    )

    static_params = dict(
#       criterion='variance',
        include_bias=True,
        missing=None,
        sampling_rate=1,
        target_comb='linear',
        target_missing=None,
        target_std=None,
    )

    def __init__(self, acts='tanh', annealing_rate=1e-06, direct=False,
                       error_func=None, hiddens=9, lasso=0, learning_rate=0.001,
                       max_iters=10, max_time=0, ridge=0, seed=0.0, std='midrange',
                       optimization='lbfgs', num_tries=10,
                       target=None, nominals=None, inputs=None):
        BaseEstimator.__init__(self, **extract_params(locals()))
        
    def fit(self, table, *args, **kwargs):
        ''' 
        Fit function for neural network

        Parameters
        ----------
        *args : dicts or two-element tuples or consecutive key/value pairs, optional
            The following types are allowed:
                * Dictionaries contain key/value pairs of parameters.  
                * Two-element tuples must contain the name of the parameter in the
                  first element and the value in the second element.  
                * Consecutive key/value pairs are also allowed.
        **kwargs : keyword arguments, optional
            These keyword arguments are the same as on the constructor.

        Examples
        --------
        >>> nn = NeuralNetwork(target='Origin',
        ...                    inputs=['MPG_City', 'MPG_Highway', 'Length',
        ...                            'Weight', 'Type', 'Cylinders'],
        ...                    nominals = ['Type', 'Cylinders', 'Origin'])
        >>> model = nn.fit(training_data)

        Returns
        -------
        :class:`NeuralNetworkModel`

        '''
        params = self.get_combined_params(*args, **kwargs)
        return self._get_super(table).fit(table, **params)


class NeuralNetworkModel(BaseModel):
    ''' NeuralNetwork trained model '''

    param_defs = NeuralNetwork.param_defs
    static_params = NeuralNetwork.static_params
