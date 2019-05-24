# -*- coding: utf-8 -*-
"""
The National Thermal Generator Performance Database.
"""
from .cli import setup_logger
from .handler import CEMS
from .clean import ParseSmoke, CleanSmoke
from .filter import Filter, PolyFit, min_hr_filter

__author__ = """Michael Rossol"""
__email__ = "michael.rossol@nrel.gov"
__version__ = "1.0.0"
