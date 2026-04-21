#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:46:24 2025

@author: steven
"""

from setuptools import setup, find_packages

setup(
    name="lattice-builder",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "lattice-builder=lattice_builder.lattice_builder_cli:main",
        ]
    },
)