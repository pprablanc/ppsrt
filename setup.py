#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='ppsrt',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pyaudio',
        'readchar',
        'numpy',
        'scipy'
        ],
    extras_require={
        #'tests': ['scipy'],
    }
)
