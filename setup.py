#!/usr/bin/env python

from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="iSeparate",
    version="0.0.1",
    description="iSeparate is a package for experimenting with Music Source Separation",
    author="Nabarun Goswami",
    author_email="nabarungoswami@mi.t.u-tokyo.ac.jp",
    packages=["iSeparate"],
    install_requires=required,
)
