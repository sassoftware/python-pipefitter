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

''' Install the SAS Pipefitter module '''

import glob
from setuptools import setup, find_packages

try:
    README = open('README.rst', 'r').read()
except:
    README = 'See README.rst'

LICENSE = 'Apache v2.0'

setup(
    zip_safe = True,
    name = 'pipefitter',
    version = '1.0.1-dev',
    description = 'SAS Pipefitter',
    long_description = README,
    author = 'SAS',
    author_email = 'Kevin.Smith@sas.com',
    url = 'http://github.com/sassoftware/python-pipefitter/',
    license = LICENSE,
    packages = find_packages(),
    install_requires = [
        'pandas >= 0.16.0',
        'saspy >= 3.6.7',
        'swat'
    ],
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
    ],
)
