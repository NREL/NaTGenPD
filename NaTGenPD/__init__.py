# -*- coding: utf-8 -*-
"""
The National Thermal Generator Performance Database.
"""
from __future__ import print_function, division, absolute_import
import os
from NaTGenPD.cli import setup_logger
from NaTGenPD.handler import CEMS, Fits
from NaTGenPD.clean import ParseSmoke, CleanSmoke
from NaTGenPD.filter import Filter, PolyFit, FitFilter
from NaTGenPD.version import __version__

__author__ = """Michael Rossol"""
__email__ = "michael.rossol@nrel.gov"
