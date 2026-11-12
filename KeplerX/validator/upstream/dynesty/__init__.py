#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dynesty is nested sampling package.
The main functionality of dynesty is performed by the
dynesty.NestedSampler and dynesty.DynamicNestedSampler
classes
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("dynesty")
except PackageNotFoundError:
    # KeplerX vendors 3.1.0 unchanged except for this fallback.
    __version__ = "3.1.0"

from .dynesty import NestedSampler, DynamicNestedSampler
from . import bounding
from . import utils
from . import pool
