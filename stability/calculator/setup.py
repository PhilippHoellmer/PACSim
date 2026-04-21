#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:46:24 2025

@author: steven
"""

from setuptools import setup, find_packages

setup(
    name="energy-calculator",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "energy-calculator=energy_calculator.energy_calculator_cli:main",
        ]
    },
)