# Copyright 2018 Rigetti & Co. Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with open(os.path.join(HERE, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def find_forest_packages():
    packages = find_packages(where='forest', exclude=('tests', 'tests.*'))
    return [f'forest.{name}' for name in packages]


setup(
    name='forest-benchmarking',
    version=find_version('forest/benchmarking', '__init__.py'),
    description='QCVV and Benchmarking',
    url='http://rigetti.com',
    author='Rigetti',
    author_email='info@rigetti.com',
    license='Apache-2.0',
    packages=find_forest_packages(),
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    install_requires=[
        'pyquil>=2.4.0',
        'numpy',
        'networkx',
        'pandas',
        'lmfit',
        'scipy',
        'sympy',
        'python-rapidjson',
        'cvxpy',
        'tqdm',
        'gitpython',
        'matplotlib'
    ],
)
