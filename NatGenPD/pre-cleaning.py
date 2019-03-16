# -*- coding: utf-8 -*-
"""
Data clustering utilities

@author: mrossol
"""
import numpy as np
import os
# import pandas as pd


class ParseSmoke:
    """
    Parse and combine SMOKE .txt files
    """
    def __init__(self, dir_path, year):
        """
        Parameters
        ----------
        dir_path : str
            Path to root directory containing SMOKE .txt files
        year : int
            Year to parse
        """
        self._smoke_files = self.get_smoke_files(dir_path, year)
        self._out_file = os.path.join(dir_path, 'SMOKE_{}.h5'.format(year))

    @staticmethod
    def get_smoke_files(dir_path, year):
        """
        Find all .txt files in dir_path associated with year

        Parameters
        ----------
        dir_path : str
            Path to root directory containing SMOKE .txt files
        year : int
            Year to parse
        """


def round_to(data, val):
    """
    round data to nearest val

    Parameters
    ----------
    data : 'ndarray', 'float'
        Input data
    perc : 'float'
        Value to round to the nearest

    Returns
    -------
    'ndarray', 'float
        Rounded data, same type as data
    """
    return data // val * val


def perc_max_filter(unit_df, perc=0.1):
    """
    Find and remove values below perc of max load or HTINPUT

    Parameters
    ----------
    unit_df : 'pd.DataFrame'
        Pandas DataFrame for individual generator unit
    perc : 'float'
        Percentage of max below which to filter

    Returns
    -------
    'ndarray'
        Indexes of bad rows
    """

    # Calculate % of max load
    load_perc = unit_df['load'] / unit_df['load'].max()
    # Calculate % of max HTINPUT
    heat_perc = unit_df['HTINPUT'] / unit_df['HTINPUT'].max()

    # Find positions for values < the perc of max load or HTINPUT
    pos = np.logical_or(heat_perc < perc, load_perc < perc)

    # Return Indexes
    return unit_df.index[pos]


def process_perc_max(smoke_df, **kwargs):
    """
    Find and remove values below perc of max load or HTINPUT

    Parameters
    ----------
    smoke_df : 'pd.DataFrame'
        Pandas DataFrame of SMOKE CEMS data
    **kwargs
        internal kwargs

    Returns
    -------
    'pd.DataFrame'
        Clean Pandas DataFrame of SMOKE CEMS data
    """
    units = np.sort(smoke_df['unit'].unique())
    unit_groups = smoke_df.groupby('unit')
    # Find indexes to be filtered
    bad_indexes = [perc_max_filter(unit_groups.get_group(unit), **kwargs)
                   for unit in units]

    return smoke_df.drop(np.hstack(bad_indexes)).reset_index(drop=True)
