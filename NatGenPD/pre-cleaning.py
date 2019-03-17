# -*- coding: utf-8 -*-
"""
Data clustering utilities

@author: mrossol
"""
import numpy as np
import os
import pandas as pd


class ParseSmoke:
    """
    Parse and combine SMOKE .txt files
    """
    SMOKE_HEADER = ["ORISID", "BLRID", "YYMMDD", "HOUR", "NOXMASS", "SO2MASS",
                    "NOXRATE", "OPTIME", "GLOAD", "SLOAD", "HTINPUT",
                    "HTINPUTMEASURE", "SO2MEASURE", "NOXMMEASURE",
                    "NOXRMEASURE", "UNITFLOW"]

    def __init__(self, dir_path, year):
        """
        Parameters
        ----------
        dir_path : str
            Path to root directory containing SMOKE .txt files
        year : int | str
            Year to parse
        """
        self._smoke_df = self.combine_smoke_files(dir_path, year)

    @property
    def smoke_df(self):
        """
        DataFrame of raw SMOKE data parsed from .txt files

        Returns
        -------
        _smoke_df : pandas.DataFrame
            DataFrame of raw SMOKE data (loaded from .txt files)
        """
        return self._smoke_df

    @staticmethod
    def get_smoke_files(dir_path, year):
        """
        Find all .txt files in dir_path associated with year

        Parameters
        ----------
        dir_path : str
            Path to root directory containing SMOKE .txt files
        year : int | str
            Year to parse

        Returns
        -------
        smoke_files : list
            List of .txt files for given year
        """
        if not isinstance(year, str):
            year = str(year)

        smoke_files = []
        for file in os.listdir(dir_path):
            if file.endswith('.txt'):
                if year in file:
                    smoke_files.append(os.path.join(dir_path, file))

        if len(smoke_files) != 12:
            missing = 12 - len(smoke_files)
            raise RuntimeError("Missing {} files for {}"
                               .format(missing, year))

        return sorted(smoke_files)

    def combine_smoke_files(self, dir_path, year):
        """
        Combine all .txt files for given year into a single DataFrame

        Parameters
        ----------
        dir_path : str
            Path to root directory containing SMOKE .txt files
        year : int | str
            Year to parse

        Returns
        -------
        smoke_df : pandas.DataFrame
            DataFrame of Smoke data for given year
        """
        smoke_files = self.get_smoke_files(dir_path, year)
        smoke_df = [pd.read_table(file, sep=',', names=self.SMOKE_HEADER)
                    for file in smoke_files]
        smoke_df = pd.concat(smoke_df)

        return smoke_df

    @staticmethod
    def unit_ids(smoke_raw):
        """
        Create unit ids from ORISID and BLRD ID

        Parameters
        ----------
        smoke_raw : pandas.DataFrame
            DataFrame of raw SMOKE data (loaded from .txt files)

        Returns
        -------
        unit_ids : pd.DataFrame
            unit ids
        """
        unit_ids = (smoke_raw['ORISID'].astype(str) + '_' +
                    smoke_raw['BLRID'].astype(str))

        return unit_ids

    @staticmethod
    def datetime(smoke_raw):
        """
        Create timestamps from YYMMDD and HOUR

        Parameters
        ----------
        smoke_raw : pandas.DataFrame
            DataFrame of raw SMOKE data (loaded from .txt files)

        Returns
        -------
        time_stamps : pandas.DateTimeIndex
            Datetime Stamps
        """
        time_stamps = pd.to_datetime(smoke_raw['YYMMDD'].astype(str),
                                     format='%y%m%d')
        time_stamps += pd.to_timedelta(smoke_raw['HOUR'], unit='h')

        return time_stamps

    @staticmethod
    def gross_load(smoke_raw):
        """
        Compute gross load from GLOAD and OPTIME

        Parameters
        ----------
        smoke_raw : pandas.DataFrame
            DataFrame of raw SMOKE data (loaded from .txt files)

        Returns
        -------
        gload : pandas.DataFrame
            gross load
        """
        gload = smoke_raw['GLOAD'] * smoke_raw['OPTIME']

        return gload

    @staticmethod
    def heat_rate(smoke_raw):
        """
        Compute heat rate in (mmBTU/MWh)

        Parameters
        ----------
        smoke_raw : pandas.DataFrame
            DataFrame of raw SMOKE data (loaded from .txt files)

        Returns
        -------
        heat_rate : pandas.DataFrame
            heat rate in mmBTU/MWh
        """
        gload = ParseSmoke.gross_load(smoke_raw)
        heat_rate = smoke_raw['HTINPUT'] / gload

        return heat_rate

    @staticmethod
    def extract_performance_vars(smoke_raw):
        """
        Extract and compute variable needed for heat rate analysis
        (unit_id, time, gload, heat_rate, OPTIME, HTINPUT, HTINPUTMEASURE)

        Parameters
        ----------
        smoke_raw : pandas.DataFrame
            DataFrame of raw SMOKE data (loaded from .txt files)

        Returns
        -------
        smoke_df : pandas.DataFrame
            DataFrame of performance variables
        """
        smoke_df = smoke_raw[['HTINPUT', 'OPTIME', 'HTINPUTMEASURE']].copy()
        smoke_df['unit_id'] = ParseSmoke.unit_ids(smoke_raw)
        smoke_df['time'] = ParseSmoke.datetime(smoke_raw)
        smoke_df['gload'] = ParseSmoke.gross_load(smoke_raw)
        smoke_df['heat_rate'] = ParseSmoke.heat_rate(smoke_raw)

        smoke_df = smoke_df[['unit_id', 'time', 'heat_rate', 'HTINPUT',
                             'gload', 'OPTIME', 'HTINPUTMEASURE']]

        return smoke_df.reset_index(drop=True)

    def save_peformance_vars(self, out_file):
        """
        Extract and compute variable needed for heat rate analysis and save
        as a .h5 file
        (unit_id, time, gload, heat_rate, OPTIME, HTINPUT, HTINPUTMEASURE)

        Parameters
        ----------
        out_file : str
            Path to output file
        """
        smoke_df = self.extract_performance_vars(self.smoke_df)
        smoke_df.to_hdf(out_file, 'smoke_df', mode='w')

    @classmethod
    def performance_vars(cls, dir_path, year, out_file=None):
        """
        Parse Smoke data from .txt, extract performance variables, and save
        to disc as a .h5 file

        Parameters
        ----------
        dir_path : str
            Path to root directory containing SMOKE .txt files
        year : int | str
            Year to parse
        out_file : str
            Path to output file, if None save to SMOKE_{year}.h5 in dir_path
        """
        smoke = cls(dir_path, year)
        if out_file is None:
            out_file = os.path.join(dir_path, 'SMOKE_{}.h5'.format(year))

        smoke.save_peformance_vars(out_file)


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
