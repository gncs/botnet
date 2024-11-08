import os
from typing import Dict

from setuptools import setup, find_packages


def readme() -> str:
    with open('README.md') as f:
        return f.read()


version_dict = {}  # type: Dict[str, str]
with open(os.path.join('botnet', '_version.py')) as file:
    exec(file.read(), version_dict)

setup(
    name='botnet',
    version=version_dict['__version__'],
    description='',
    long_description=readme(),
    classifiers=['Programming Language :: Python :: 3.7'],
    python_requires='>=3.7',
    packages=find_packages(),
    install_requires=['e3nn', 'torch'],
    zip_safe=False,
    test_suite='pytest',
    tests_require=['pytest'],
)
