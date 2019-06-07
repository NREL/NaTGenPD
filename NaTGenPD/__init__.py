# -*- coding: utf-8 -*-
"""
The National Thermal Generator Performance Database.
"""
from NaTGenPD.cli import setup_logger
from NaTGenPD.handler import CEMS
from NaTGenPD.clean import ParseSmoke, CleanSmoke
from NaTGenPD.filter import Filter, PolyFit, min_hr_filter
from NaTGenPD.version import __version__

__author__ = """Michael Rossol"""
__email__ = "michael.rossol@nrel.gov"
