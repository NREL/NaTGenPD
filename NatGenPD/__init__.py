"""
The National Thermal Generator Performance Database.
"""
from .cli import setup_logger
from .handler import CEMS
from . precleaning import ParseSmoke, CleanSmoke

__author__ = """Michael Rossol"""
__email__ = "michael.rossol@nrel.gov"
__version__ = "1.0.0"
