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
Parameter management utilities

'''

from __future__ import print_function, division, absolute_import, unicode_literals

import collections
import copy
import math
import numbers
import re
import six
import sys
import textwrap
import weakref
from six.moves.urllib.parse import urlparse

# Parameter options
READ_ONLY_PARAMETER = 1

def splat(obj):
    ''' Ensure that the object is always a list '''
    if obj is None:
        return None
    if isinstance(obj, (list, set, tuple)):
        return list(obj)
    return [obj]

def extract_params(loc):
    ''' Extract parameters from locals '''
    loc = loc.copy()
    loc.pop('self')
    return loc

def param_def(default=None, validator=None, doc=None, options=0, name=None, owner=None):
    ''' Construct a complete parameter definition '''
    return dict(default=default, validator=validator, doc=doc,
                options=options, name=name, owner=owner)

def get_params_doc(obj):
    '''
    Extract the parameter documentation from the given object 

    Parameters
    ----------
    obj : any
        The object to get the docstring from

    Returns
    -------
    dict
        The keys are the parameter names.  The values are the documentation
        string associated with that name.

    '''
    params_header = r'\n\s*Parameters\s*-+\s*\n'
    any_header = r'\n\s*[A-Z]\w+\s*-+\s*\n'
    
    doc = getattr(obj, '__doc__', '')
    if not doc or not re.search(params_header, doc):
        return dict()

    params = re.split(params_header, doc)[1]
    params = re.split(any_header, params)[0].rstrip()
    params = textwrap.dedent(params)
    params = re.split(r'^(\w+)\s*:.*$\n', params, flags=re.M)[1:]

    dociter = iter(params)
    params = [(x, textwrap.dedent(y).rstrip()) for x, y in zip(dociter, dociter)]

    return dict(params)


class param_property(object):
    ''' Accessor for parameters '''

    def __init__(self, name, doc):
        self.name = name
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        ''' Return the parameter object '''
        if obj is None:
            return self
        return obj.params.get_parameter(self.name)

    def __set__(self, obj, value):
        ''' Set a new parameter value '''
        obj.params[self.name] = value


@six.python_2_unicode_compatible
class ParameterManager(object):
    ''' Manage object parameters '''

    param_defs = dict()
    static_params = dict()

    def __init__(self, **kwargs):
        self.params = ParameterDict()

        params_doc = get_params_doc(type(self))

        # Create parameters from arguments
        for key, value in type(self).param_defs.items():
            if isinstance(value, Parameter):
                value = value.copy()
                value._name = key
                value._owner = self
                if not value.__doc__:
                    value.__doc__ = params_doc.get(key, '')
                out = self.params.add_parameter(value)
            else:
                value = value.copy()
                value['name'] = key
                value['owner'] = self
                if not value.get('doc', None):
                    value['doc'] = params_doc.get(key, '')
                out = self.params.add_parameter(**value)

            # Add a property to the class for each parameter
            setattr(type(self), out._name, param_property(out._name, out.__doc__))

        self.set_params(**kwargs)

    def __str__(self):
        params = []
        for key, value in sorted(self.params.items()):
            params.append('%s=%s' % (key, repr(value)))
        return '%s(%s)' % (type(self).__name__, ', '.join(params))

    def __repr__(self):
        return str(self)

    def _set_options(self, options):
        self.params._set_options(options)

    def set_params(self, *args, **kwargs):
        '''
        Set one or more parameters 

        '''
        self.params.set(*args, **kwargs)

    set_param = set_params

    def get_params(self, *names):
        '''
        Return a copy of the requested parameters

        Parameters
        ----------
        names : strings, optional
            The names of the parameters to return

        Returns
        -------
        dict

        '''
        if names:
            out = {}
            for name in names:
                out[name] = self.params[name]
            return out
        return self.params.to_dict()

    get_param = get_params

    def has_param(self, name):
        '''
        Does the parameter exist?

        Parameters
        ----------
        name : string
            Name of the parameter

        Returns
        -------
        boolean

        '''
        return name in self.params

    def get_combined_params(self, *args, **kwargs):
        '''
        Merge all parameters and verify that they valid

        This method allows a mixed bag of parameters in a dictionary or list
        of arguments to be combined with the parameters of `self`, but only if
        the key values are generic or Parameters belonging to `self`.

        Examples
        --------
        >>> params = {pm1.int_value: 100, pm1.float_value: 1.23}

        >>> pm1.get_combined_params(params)
        {'int_value': 100, 'float_value': 1.23}

        '''
        out = self.params.copy()

        if kwargs:
            argiter = iter(list(args) + [kwargs])
        else:
            argiter = iter(list(args))

        specific_params = {}

        for arg in argiter:
            if isinstance(arg, ParameterManager):
                raise TypeError('ParameterManager arguments are not valid')
            elif hasattr(arg, 'items') and callable(arg.items):
                for key, value in arg.items():
                    if isinstance(key, Parameter):
                        specific_params[key._name] = value
                    else:
                        out[key] = value
            elif isinstance(arg, (list, tuple)):
                if len(arg) < 2:
                    raise ValueError('Parameter for "%s" is missing a value')
                if len(arg) > 2:
                    raise ValueError('Too many elements in parameter tuple: %s' % (arg,))
                if not isinstance(arg[0], six.string_types) and \
                        not isinstance(arg[0], Parameter):
                    raise TypeError('Key is not a string or Parameter: %s' % arg[0])
                key, value = arg
                if isinstance(key, Parameter):
                    specific_params[key._name] = value
                else:
                    out[key] = value
            elif isinstance(arg, six.string_types) or isinstance(arg, Parameter):
                try:
                    value = next(argiter)
                except StopIteration:
                    raise ValueError('Parameter "%s" is missing a value')
                if isinstance(arg, Parameter):
                    specific_params[arg._name] = value
                else:
                    out[arg] = value
            else:
                raise TypeError('Unknown type for parameter: %s' % arg)

        for key, value in specific_params.items():
            out[key] = value

        return out.to_dict()

    def get_filtered_params(self, *args, **kwargs):
        '''
        Merge parameters that keys that belong to `self`

        This method allows a mixed bag of parameters in a dictionary or list
        of arguments to be combined with the parameters of `self`, but only if
        the key values are generic or Parameters belonging to `self`.

        Examples
        --------
        >>> params = {pm1.int_value: 100, pm1.float_value: 1.23,
                      pm2.int_value: 101, pm2.str_value: 'foo'}

        >>> pm1.get_filtered_params(params)
        {'int_value': 100, 'float_value': 1.23}

        >>> pm2.get_filtered_params(params)
        {'int_value': 101, 'str_value': 'foo'}

        '''
        out = self.params.copy()

        if kwargs:
            argiter = iter(list(args) + [kwargs])
        else:
            argiter = iter(list(args))

        specific_params = {}

        for arg in argiter:
            if isinstance(arg, ParameterManager):
                pass
            elif hasattr(arg, 'items') and callable(arg.items):
                for key, value in arg.items():
                    if isinstance(key, Parameter):
                        if key._owner is self:
                            specific_params[key._name] = value
                    elif key in self.params:
                        out[key] = value
            elif isinstance(arg, (list, tuple)):
                if len(arg) < 2:
                    raise ValueError('Parameter for "%s" is missing a value')
                if len(arg) > 2:
                    raise ValueError('Too many elements in parameter tuple: %s' % (arg,))
                if not isinstance(arg[0], six.string_types) and \
                        not isinstance(arg[0], Parameter):
                    raise TypeError('Key is not a string or Parameter: %s' % arg[0])
                key, value = arg
                if isinstance(key, Parameter):
                    if key._owner is self:
                        specific_params[key._name] = value
                elif key in self.params:
                    out[key] = value
            elif isinstance(arg, six.string_types) or isinstance(arg, Parameter):
                try:
                    value = next(argiter)
                except StopIteration:
                    raise ValueError('Parameter "%s" is missing a value')
                if isinstance(arg, Parameter):
                    if arg._owner is self:
                        specific_params[arg._name] = value
                else:
                    if arg in self.params:
                        out[arg] = value
            else:
                raise TypeError('Unknown type for parameter: %s' % arg)

        for key, value in specific_params.items():
            out[key] = value

        return out.to_dict()


def check_variable(value, pattern=None, valid_values=None, normalize=False,
                   allow_none=True):
    '''
    Validate a variable name

    Parameters
    ----------
    value : string
        Value to validate
    pattern : regular expression string, optional
        A regular expression used to validate name value
    valid_values : list of strings, optional
        List of the only possible values
    normalize : boolean, optional
        Should the name be normalized (lower-cased)?
    allow_none : boolean, optional
        Should a None value be allowed?

    Returns
    -------
    string
        The validated name value

    '''
    if allow_none and value is None:
        return None

    try:
        return check_string(value, pattern=pattern, valid_values=valid_values,
                            normalize=normalize)
    except ValueError:
        raise ValueError('Value is not a valid variable name')


def check_variable_list(values, pattern=None, valid_values=None,
                        normalize=False, allow_empty=True):
    '''
    Validate a list of variable names

    Parameters
    ----------
    value : string or list of strings
        Value to validate
    pattern : regular expression string, optional
        A regular expression used to validate name values
    valid_values : list of strings, optional
        List of the only possible values
    normalize : boolean, optional
        Should the names be normalized (lower-cased)?
    allow_empty : boolean, optional
        Should an empty list be allowed?

    Returns
    -------
    list of strings
        The validated name values

    '''
    if values is None:
        if not allow_empty:
            raise ValueError('The variable list is empty')
        return []

    if not isinstance(values, (list, tuple, set)):
        values = [values]

    values = list(values)

    for i, item in enumerate(values):
        try:
            values[i] = check_string(item, pattern=pattern,
                                     valid_values=valid_values,
                                     normalize=normalize)
        except ValueError:
            raise ValueError('%s is not a valid variable name' % item)

    if not allow_empty and not values:
        raise ValueError('The variable list is empty')

    return values


def check_int(value, minimum=None, maximum=None, exclusive_minimum=False,
              exclusive_maximum=False, multiple_of=None, allow_none=False):
    '''
    Validate an integer value

    Parameters
    ----------
    value : int or float
        Value to validate
    minimum : int, optional
        The minimum value allowed
    maximum : int, optional
        The maximum value allowed
    exclusive_minimum : boolean, optional
        Should the minimum value be excluded as an endpoint?
    exclusive_maximum : boolean, optional
        Should the maximum value be excluded as an endpoint?
    multiple_of : int, optional
        If specified, the value must be a multple of it in order for
        the value to be considered valid.
    allow_none : boolean, optional
        Should a None value be allowed?

    Returns
    -------
    int
        The validated integer value

    '''
    if allow_none and value is None:
        return value

    out = int(value)

    if minimum is not None:
        if out < minimum:
            raise ValueError('%s is smaller than the minimum value of %s' %
                             (out, minimum))
        if exclusive_minimum and out == minimum:
            raise ValueError('%s is equal to the exclusive nimum value of %s' %
                             (out, minimum))

    if maximum is not None:
        if out > maximum:
            raise ValueError('%s is larger than the maximum value of %s' %
                             (out, maximum))
        if exclusive_maximum and out == maximum:
            raise ValueError('%s is equal to the exclusive maximum value of %s' %
                             (out, maximum))

    if multiple_of is not None:
        if (out % int(multiple_of)) != 0:
            raise ValueError('%s is not a multiple of %s' % (out, multiple_of))

    return out


def check_int_list(values, minimum=None, maximum=None, exclusive_minimum=False,
                   exclusive_maximum=False, multiple_of=None, allow_empty=False,
                   allow_none=False):
    '''
    Validate a list of integer values

    Parameters
    ----------
    value : list-of-ints or list-of-floats
        Value to validate
    minimum : int, optional
        The minimum value allowed
    maximum : int, optional
        The maximum value allowed
    exclusive_minimum : boolean, optional
        Should the minimum value be excluded as an endpoint?
    exclusive_maximum : boolean, optional
        Should the maximum value be excluded as an endpoint?
    multiple_of : int, optional
        If specified, the value must be a multple of it in order for
        the value to be considered valid.
    allow_empty : boolean, optional
        Should an empty list be allowed?
    allow_none : boolean, optional
        Should a None value be allowed?

    Returns
    -------
    list-of-ints
        The validated integer list values

    '''
    if values is None:
        if allow_none:
            return
        if not allow_empty:
            raise ValueError('The integer list is empty')
        return []

    if not isinstance(values, (list, tuple, set)):
        values = [values]

    values = list(values)

    for i, item in enumerate(values):
        try:
            values[i] = check_int(item, minimum=minimum, maximum=maximum,
                                  exclusive_minimum=exclusive_minimum,
                                  exclusive_maximum=exclusive_maximum,
                                  multiple_of=multiple_of)
        except ValueError:
            raise ValueError('%s is not a valid integer value' % item)

    if not allow_empty and not values:
        raise ValueError('The integer list is empty')

    return values


def check_float(value, minimum=None, maximum=None, exclusive_minimum=False,
                exclusive_maximum=False, multiple_of=None, allow_none=False):
    '''
    Validate a floating point value

    Parameters
    ----------
    value : int or float
        Value to validate
    minimum : int or float, optional
        The minimum value allowed
    maximum : int or float, optional
        The maximum value allowed
    exclusive_minimum : boolean, optional
        Should the minimum value be excluded as an endpoint?
    exclusive_maximum : boolean, optional
        Should the maximum value be excluded as an endpoint?
    multiple_of : int or float, optional
        If specified, the value must be a multple of it in order for
        the value to be considered valid.
    allow_none : boolean, optional
        Should a None value be allowed?

    Returns
    -------
    float
        The validated floating point value

    '''
    if allow_none and value is None:
        return value

    out = float(value)

    if minimum is not None:
        if out < minimum:
            raise ValueError('%s is smaller than the minimum value of %s' %
                             (out, minimum))
        if exclusive_minimum and out == minimum:
            raise ValueError('%s is equal to the exclusive nimum value of %s' %
                             (out, minimum))

    if maximum is not None:
        if out > maximum:
            raise ValueError('%s is larger than the maximum value of %s' %
                             (out, maximum))
        if exclusive_maximum and out == maximum:
            raise ValueError('%s is equal to the exclusive maximum value of %s' %
                             (out, maximum))

    if multiple_of is not None:
        if (out % int(multiple_of)) != 0:
            raise ValueError('%s is not a multiple of %s' % (out, multiple_of))

    return out


def check_float_list(values, minimum=None, maximum=None, exclusive_minimum=False,
                     exclusive_maximum=False, multiple_of=None, allow_empty=False,
                     allow_none=False):
    '''
    Validate a list of floating point values

    Parameters
    ----------
    value : int or float
        Value to validate
    minimum : int or float, optional
        The minimum value allowed
    maximum : int or float, optional
        The maximum value allowed
    exclusive_minimum : boolean, optional
        Should the minimum value be excluded as an endpoint?
    exclusive_maximum : boolean, optional
        Should the maximum value be excluded as an endpoint?
    multiple_of : int or float, optional
        If specified, the value must be a multple of it in order for
        the value to be considered valid.
    allow_empty : boolean, optional
        Should an empty list be allowed?
    allow_none : boolean, optional
        Should a None value be allowed?

    Returns
    -------
    list-of-floats
        The validated floating point value list

    '''
    if values is None:
        if allow_none:
            return
        if not allow_empty:
            raise ValueError('The float list is empty')
        return []

    if not isinstance(values, (list, tuple, set)):
        values = [values]

    values = list(values)

    for i, item in enumerate(values):
        try:
            values[i] = check_float(item, minimum=minimum, maximum=maximum,
                                    exclusive_minimum=exclusive_minimum,
                                    exclusive_maximum=exclusive_maximum,
                                    multiple_of=multiple_of)
        except ValueError:
            raise ValueError('%s is not a valid float value' % item)

    if not allow_empty and not values:
        raise ValueError('The float list is empty')

    return values


def check_number_or_iter(value, minimum=None, maximum=None, exclusive_minimum=False,
                         exclusive_maximum=False, multiple_of=None, allow_none=False,
                         minimum_int=None, maximum_int=None, minimum_float=None,
                         maximum_float=None):
    if allow_none and value is None:
        return value
    if isinstance(value, six.string_types):
        raise TypeError('Type must be numeric or iterable')
    elif isinstance(value, collections.Iterable):
        return value
    return check_number(value, minimum=minimum, maximum=maximum,
                        exclusive_minimum=exclusive_minimum,
                        multiple_of=multiple_of, allow_none=allow_none,
                        minimum_int=minimum_int, maximum_int=maximum_int,
                        minimum_float=minimum_float, maximum_float=maximum_float)


def check_number(value, minimum=None, maximum=None, exclusive_minimum=False,
                 exclusive_maximum=False, multiple_of=None, allow_none=False,
                 minimum_int=None, maximum_int=None, minimum_float=None,
                 maximum_float=None):
    if allow_none and value is None:
        return value
    if isinstance(value, numbers.Integral):
        func = check_int
        if minimum_int is not None:
            minimum = minimum_int
        if maximum_int is not None:
            maximum = maximum_int
    else:
        func = check_float
        if minimum_float is not None:
            minimum = minimum_float
        if maximum_float is not None:
            maximum = maximum_float
    return func(value, minimum=minimum, maximum=maximum,
                exclusive_minimum=exclusive_minimum,
                exclusive_maximum=exclusive_maximum,
                multiple_of=multiple_of)


def check_number_list(values, minimum=None, maximum=None, exclusive_minimum=False,
                      exclusive_maximum=False, multiple_of=None, allow_empty=False,
                      allow_none=False, minimum_int=None, maximum_int=None,
                      minimum_float=None, maximum_float=None):
    if values is None:
        if allow_none:
            return
        if not allow_empty:
            raise ValueError('The number list is empty')
        return []

    if not isinstance(values, (list, tuple, set)):
        values = [values]

    values = list(values)

    for i, item in enumerate(values):
        try:
            values[i] = check_number(item, minimum=minimum, maximum=maximum,
                                     exclusive_minimum=exclusive_minimum,
                                     exclusive_maximum=exclusive_maximum,
                                     multiple_of=multiple_of,
                                     minimum_int=minimum_int, maximum_int=maximum_int,
                                     minimum_float=minimum_float,
                                     maximum_float=maximum_float)
        except ValueError:
            raise ValueError('%s is not a valid number value' % item)

    if not allow_empty and not values:
        raise ValueError('The number list is empty')

    return values


def check_boolean(value):
    '''
    Validate a boolean value

    Parameters
    ----------
    value : int or boolean
        The value to validate.  If specified as an integer, it must
        be either 0 for False or 1 for True.

    Returns
    -------
    boolean
        The validated boolean

    '''
    if not isinstance(value, bool) and not isinstance(value, six.integer_types):
        raise TypeError('Boolean values must be bools or integers')

    if value is False or value is True:
        return value

    if isinstance(value, six.integer_types):
        if value == 1:
            return True
        if value == 0:
            return False

    raise ValueError('%s is not a boolean or proper integer value')


def check_string(value, pattern=None, max_length=None, min_length=None,
                 valid_values=None, normalize=False, allow_none=False):
    '''
    Validate a string value

    Parameters
    ----------
    value : string
        The value to validate
    pattern : regular expression string, optional
        A regular expression used to validate string values
    max_length : int, optional
        The maximum length of the string
    min_length : int, optional
        The minimum length of the string
    valid_values : list of strings, optional
        List of the only possible values
    normalize : boolean, optional
        Should the strings be normalized (lower-cased)?
    allow_none : boolean, optional
        Should a None value be allowed?

    Returns
    -------
    string
        The validated string value

    '''
    if allow_none and value is None:
        return value

    if not isinstance(value, six.string_types):
        raise TypeError('%s is not a string value' % value)

    out = six.text_type(value)

    if normalize:
        out = out.lower()

    if max_length is not None and len(out) > max_length:
        raise ValueError('%s is longer than the maximum length of %s' %
                         (out, max_length))

    if min_length is not None and len(out) < min_length:
        raise ValueError('%s is shorter than the minimum length of %s' %
                         (out, min_length))

    if pattern is not None:
        if isinstance(pattern, six.string_types):
            if not re.search(pattern, out):
                raise ValueError('"%s" does not match pattern "%s"' % (out, pattern))
        elif not pattern.search(out):
            raise ValueError('"%s" does not match pattern %s' % (out, pattern))

    if valid_values is not None and out not in valid_values:
        raise ValueError('%s is not one of the possible values: %s' %
                         (out, ', '.join(valid_values)))

    return out


def check_url(value, pattern=None, max_length=None, min_length=None, valid_values=None):
    '''
    Validate a URL value

    Parameters
    ----------
    value : any
        The value to validate.  This value will be cast to a string
        and converted to unicode.
    pattern : regular expression string, optional
        A regular expression used to validate string values
    max_length : int, optional
        The maximum length of the string
    min_length : int, optional
        The minimum length of the string
    valid_values : list of strings, optional
        List of the only possible values

    Returns
    -------
    string
        The validated URL value

    '''
    if not isinstance(value, six.string_types):
        raise TypeError('%s is not a string value')

    out = check_string(six.text_type(value), pattern=pattern, max_length=max_length,
                       min_length=min_length, valid_values=valid_values)

    urlparse(out)

    return out


@six.python_2_unicode_compatible
class Parameter(object):
# 
# NOTE: Commented out so that the descriptor docstring will work 
#       in the IPython ? operator.
#
#   '''
#   Generic object parameter

#   Parameters
#   ----------
#   owner : instance
#       The instance object that the parameter belongs to
#   name : string
#       The name of the option
#   default : any
#       The default value of the option
#   validator : callable
#       A callable object that validates the option value and returns
#       the validated value.
#   doc : string
#       The documentation string for the option
#   options : int
#       READ_ONLY_PARAMETER 
#
#   Returns
#   -------
#   Parameter object

#   '''

    param_ids = {}

    def __init__(self, owner, name, default, validator=None, doc=None, options=0):
        if validator is None:
            validator = lambda x: x
        self._owner = owner
        self._name = name
        self._validator = validator
        self._default = validator(default)
        self._value = self._default
        self._options = options
        self._is_set = False
        self.__doc__ = doc and doc.rstrip() or ''
        type(self).param_ids.setdefault(self._name,
                                        '1%0.4d' % len(type(self).param_ids))

    def __str__(self):
        return six.text_type(self._value)

    def __repr__(self):
        return repr(self._value)

    def set_value(self, value):
        '''
        Set the value of the parameter

        Parameters
        ----------
        value : any
           The value to set

        '''
        if self._options & READ_ONLY_PARAMETER:
            raise RuntimeError('%s is a read-only parameter' % self._name)
        self._is_set = True
        self._value = self._validator(value)

    def get_value(self):
        ''' Return the value of the option '''
        return self._value

    def get_default(self):
        ''' Return the default value of the option '''
        return self._default

    def is_default(self):
        ''' Is the current value set to the default?  '''
        return not self._is_set

    def is_set(self):
        ''' Has the parameter been set? '''
        return self._is_set

    def copy(self):
        ''' Return a copy of the parameter '''
        out = type(self)(self._owner, self._name, self._default, self._validator, '')
        out._value = self._value
        out._is_set = self._is_set
        out.__doc__ = self.__doc__
        return out

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo=None):
        return self.copy()

    def __hash__(self):
        return int(type(self).param_ids[self._name] + str(id(self._owner)))

    # Numeric operations

    def __add__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        out = self.copy()
        out.set_value(out._value + other)
        return out

    def __sub__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        out = self.copy()
        out.set_value(out._value - other)
        return out

    def __mul__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        out = self.copy()
        out.set_value(out._value * other)
        return out

    def __truediv__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        out = self.copy()
        out.set_value(out._value / other)
        return out

    def __floordiv__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        out = self.copy()
        out.set_value(out._value // other)
        return out

    def __mod__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        out = self.copy()
        out.set_value(out._value % other)
        return out

# Result is a tuple, not a single value
#   def __divmod__(self, other):
#       if isinstance(other, Parameter):
#           other = other._value
#       out = self.copy()
#       out.set_value(divmod(out._value, other))
#       return out

    def __pow__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        out = self.copy()
        out.set_value(out._value ** other)
        return out

    def __lshift__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        out = self.copy()
        out.set_value(out._value << other)
        return out

    def __rshift__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        out = self.copy()
        out.set_value(out._value >> other)
        return out

    def __and__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        out = self.copy()
        out.set_value(out._value & other)
        return out

    def __xor__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        out = self.copy()
        out.set_value(out._value ^ other)
        return out

    def __or__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        out = self.copy()
        out.set_value(out._value | other)
        return out

    def __radd__(self, other):
        if isinstance(other, Parameter):
            return other.__add__(self._value)
        out = self.copy()
        out.set_value(other + out._value)
        return out

    def __rsub__(self, other):
        if isinstance(other, Parameter):
            return other.__sub__(self._value)
        out = self.copy()
        out.set_value(other - out._value)
        return out

    def __rmul__(self, other):
        if isinstance(other, Parameter):
            return other.__mul__(self._value)
        out = self.copy()
        out.set_value(other * out._value)
        return out

    def __rtruediv__(self, other):
        if isinstance(other, Parameter):
            return other.__truediv__(self._value)
        out = self.copy()
        out.set_value(other / out._value)
        return out

    def __rfloordiv__(self, other):
        if isinstance(other, Parameter):
            return other.__floordiv__(self._value)
        out = self.copy()
        out.set_value(other // out._value)
        return out

    def __rmod__(self, other):
        if isinstance(other, Parameter):
            return other.__mod__(self._value)
        out = self.copy()
        out.set_value(other % out._value)
        return out

# Result is a tuple, not a single value
#   def __rdivmod__(self, other):
#       if isinstance(other, Parameter):
#           return other.__divmod__(self._value)
#       out = self.copy()
#       out.set_value(divmod(other, out._value))
#       return out

    def __rpow__(self, other):
        if isinstance(other, Parameter):
            return other.__pow__(self._value)
        out = self.copy()
        out.set_value(other ** out._value)
        return out

    def __rlshift__(self, other):
        if isinstance(other, Parameter):
            return other.__lshift__(self._value)
        out = self.copy()
        out.set_value(other << out._value)
        return out

    def __rrshift__(self, other):
        if isinstance(other, Parameter):
            return other.__rshift__(self._value)
        out = self.copy()
        out.set_value(other >> out._value)
        return out

    def __rand__(self, other):
        if isinstance(other, Parameter):
            return other.__and__(self._value)
        out = self.copy()
        out.set_value(other & out._value)
        return out

    def __rxor__(self, other):
        if isinstance(other, Parameter):
            return other.__xor__(self._value)
        out = self.copy()
        out.set_value(other ^ out._value)
        return out

    def __ror__(self, other):
        if isinstance(other, Parameter):
            return other.__or__(self._value)
        out = self.copy()
        out.set_value(other | out._value)
        return out

    def __iadd__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        self.set_value(self._value + other)
        return self

    def __isub__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        self.set_value(self._value - other)
        return self

    def __imul__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        self.set_value(self._value * other)
        return self

    def __itruediv__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        self.set_value(self._value / other)
        return self

    def __ifloordiv__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        self.set_value(self._value // other)
        return self

    def __imod__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        self.set_value(self._value % other)
        return self

    def __ipow__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        self.set_value(self._value ** other)
        return self

    def __ilshift__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        self.set_value(self._value << other)
        return self

    def __irshift__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        self.set_value(self._value >> other)
        return self

    def __iand__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        self.set_value(self._value & other)
        return self

    def __ixor__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        self.set_value(self._value ^ other)
        return self

    def __ior__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        self.set_value(self._value | other)
        return self

    def __neg__(self):
        out = self.copy()
        out.set_value(-out._value)
        return out

    def __pos__(self):
        out = self.copy()
        out.set_value(+out._value)
        return out

    def __abs__(self):
        out = self.copy()
        out.set_value(abs(out._value))
        return out

    def __invert__(self):
        out = self.copy()
        out.set_value(~out._value)
        return out

#   def __complex__(self):
#       out = self.copy()
#       out.set_value(complex(out._value))
#       return out

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)

    def __round__(self, n=0):
        out = self.copy()
        out.set_value(round(out._value, n))
        return out

    def __ceil__(self):
        out = self.copy()
        out.set_value(math.ceil(out._value))
        return out

    def __floor__(self):
        out = self.copy()
        out.set_value(math.floor(out._value))
        return out

    def __trunc__(self):
        out = self.copy()
        out.set_value(math.trunc(out._value))
        return out

    # Comparison operators

    def __eq__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        return self._value == other 

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        return self._value < other

    def __le__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        return self._value <= other

    def __gt__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        return self._value > other

    def __ge__(self, other):
        if isinstance(other, Parameter):
            other = other._value
        return self._value >= other

    def __bool__(self):
        return bool(self._value)


@six.python_2_unicode_compatible
class ParameterDict(object):
    ''' Dictionary-like object that validates key/value pairs '''

    def __init__(self, *args, **kwargs):
       self._params = {} 
       self.add_parameter(*args, **kwargs)

    def __str__(self):
        return six.text_type(dict(self.items()))

    def __repr__(self):
        return repr(dict(self.items()))

    def _set_options(self, options):
        for value in self._params.values():
            value._options |= options

    def add_parameter(self, *args, **kwargs):
        out = []

        for item in args:
            if isinstance(item, Parameter):
                out.append(item)
                self._params[item._name] = item
            else:
                raise TypeError('%s is not a Parameter instance' % item)

        if kwargs:
            item = Parameter(**kwargs)
            self._params[item._name] = item
            out.append(item)

        if len(out) > 1:
            return out

        if out:
            return out[0]

    def del_parameter(self, *keys):
        for key in keys:
            self._params.pop(key, None) 

    def get_parameter(self, key, *default):
        if isinstance(key, Parameter):
            key = key._name
        try:
            return self._params[key]
        except KeyError:
            if default:
                return default[0]
            raise

    def describe_parameter(self, *keys, **kwargs):
        output = kwargs.get('output', sys.stdout)
        last = len(keys) - 1
        indent = textwrap.TextWrapper(initial_indent='    ', subsequent_indent='    ').fill
        for i, key in enumerate(keys):
            param = self._params[key]
            output.write(param._name)
            output.write('\n')
            output.write(indent(param.__doc__))
            output.write('\n')
            output.write('    [Current: %s] [Default: %s]' %
                         (repr(param._value), repr(param._default)))
            if i != last:
                output.write('\n\n')

    def get(self, key, *default):
        if isinstance(key, Parameter):
            key = key._name
        try:
            return self._params[key].get_value()
        except KeyError:
            if default:
                return default[0]
            raise

    def set(self, *args, **kwargs):
        '''
        Set one or more parameters

        '''
        argiter = iter(args)
        for arg in argiter:
            if isinstance(arg, ParameterManager):
                self.update(dict(arg.params.items()))
            elif hasattr(arg, 'items') and callable(arg.items):
                self.update(dict(arg.items()))
            elif isinstance(arg, Parameter):
                self[arg._name] = arg
            elif isinstance(arg, (list, tuple)):
                if len(arg) < 2:
                    raise ValueError('Parameter for "%s" is missing a value')
                if len(arg) > 2:
                    raise ValueError('Too many elements in parameter tuple: %s' % (arg,))
                if not isinstance(arg[0], six.string_types):
                    raise TypeError('Key is not a string: %s' % arg[0])
                self[arg[0]] = arg[1]
            elif isinstance(arg, six.string_types):
                try:
                    self[arg] = next(argiter)
                except StopIteration:
                    raise ValueError('Parameter "%s" is missing a value')
            else:
                raise TypeError('Unknown type for parameter: %s' % arg)

        self.update(kwargs)

    def __getitem__(self, key):
        if isinstance(key, Parameter):
            key = key._name
        return self._params[key].get_value()

    def __setitem__(self, key, value):
        if isinstance(key, Parameter):
            key = key._name
        if not isinstance(key, six.string_types):
            raise TypeError('Key values must be strings: %s' % key)
        if key not in self._params:
            raise KeyError('%s is not a valid parameter key for this object' % key)
        if key in self._params and self._params[key]._options & READ_ONLY_PARAMETER:
            raise RuntimeError('%s is a read-only parameter' % key)
        if isinstance(value, Parameter):
            value = value._value
        out = self._params[key].copy()
        out.set_value(value)
        self._params[key] = out

    def update(self, *args, **kwargs):
        for item in (list(args) + [kwargs]):
            if hasattr(item, 'items') and callable(item.items):
                for key, value in item.items():
                    self[key] = value
            elif isinstance(item, Parameter):
                self[item._name] = item._value
            else:
                self[item[0]] = item[1]

    def keys(self):
        return list(self._params.keys())

    def values(self):
        return [x.get_value() for x in self._params.values()]

    def items(self):
        return [(x, self._params[x]) for x in self._params.keys()]

    def __iter__(self):
        for key in self.keys():
            yield key

    def __contains__(self, key):
        return key in self._params.keys() 

    def __len__(self):
        return len(self._params.keys())

    def copy(self):
        out = type(self)()
        for value in self._params.values():
            out.add_parameter(value.copy())
        return out

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo=None):
        return self.copy()

    def __eq__(self, other):
        if hasattr(other, 'items') and callable(other.items):
            newdict = {}
            for key, value in other.items():
                if isinstance(key, Parameter):
                    key = key._name 
                newdict[key] = value
            other = newdict
        return dict(self.items()) == other

    def to_dict(self):
        return {k: v._value for k, v in self.items()} 
