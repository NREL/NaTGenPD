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
        self._smoke_raw = self.combine_smoke_files(dir_path, year)

    @property
    def smoke_raw(self):
        """
        DataFrame of raw SMOKE data parsed from .txt files

        Returns
        -------
        _smoke_df : pandas.DataFrame
            DataFrame of raw SMOKE data (loaded from .txt files)
        """
        return self._smoke_raw

    @property
    def smoke_df(self):
        """
        DataFrame of performance variables derived from SMOKE data

        Returns
        -------
        performance_df : pandas.DataFrame
            DataFrame of performance varialbes derived from SMOKE data
            (unit_id, time, gload, heat_rate, OPTIME, HTINPUT, HTINPUTMEASURE)
        """
        smoke_df = self.extract_performance_vars(self.smoke_raw)
        return smoke_df

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
    def create_unit_ids(smoke_raw):
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
    def create_datetime(smoke_raw):
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
    def calc_gross_load(smoke_raw):
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
    def calc_heat_rate(smoke_raw):
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
        gload = ParseSmoke.calc_gross_load(smoke_raw)
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
            DataFrame of performance variables from SMOKE data
        """
        smoke_df = smoke_raw[['HTINPUT', 'OPTIME', 'HTINPUTMEASURE']].copy()
        smoke_df['unit_id'] = ParseSmoke.create_unit_ids(smoke_raw)
        smoke_df['time'] = ParseSmoke.create_datetime(smoke_raw)
        smoke_df['gload'] = ParseSmoke.calc_gross_load(smoke_raw)
        smoke_df['heat_rate'] = ParseSmoke.calc_heat_rate(smoke_raw)

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
        smoke_df = self.smoke_df
        smoke_df.to_hdf(out_file, 'smoke_df', mode='w')

    @classmethod
    def performance_vars(cls, dir_path, year, save=True):
        """
        Parse Smoke data from .txt, extract performance variables, and save
        to disc as a .h5 file

        Parameters
        ----------
        dir_path : str
            Path to root directory containing SMOKE .txt files
        year : int | str
            Year to parse
        save : bool
            Flag to save data to .h5 file in dir_path

        Returns
        -------
        smoke_df : pandas.DataFrame
            DataFrame of performance variables from SMOKE data
        """
        smoke = cls(dir_path, year)
        if save:
            out_file = os.path.join(dir_path, 'SMOKE_{}.h5'.format(year))
            smoke.save_peformance_vars(out_file)

        return smoke.smoke_df


class ParseUnitInfo:
    """
    Extract Unit locational (coordinate, state, regions) and
    type (fuel, technology)
    """
    COLS = {' Facility ID (ORISPL)': 'ORISID',
            ' Unit ID': 'BLRID',
            ' Facility Latitude': 'latitude',
            ' Facility Longitude': 'longitude',
            'State': 'state',
            ' EPA Region': 'EPA_region',
            ' NERC Region': 'NERC_region',
            ' Unit Type': 'unit_type',
            ' Fuel Type (Primary)': 'fuel_type'}

    def __init__(self, unit_attrs_path):
        """
        Parameters
        ----------
        unit_attrs_path : str
            Path to .csv containing facility (unit) attributes
        """
        self._unit_attrs = self._parse_csv(unit_attrs_path)

    @property
    def unit_attrs(self):
        """
        Unit attributes parsed from .csv

        Returns
        ----------
        _unit_attrs: pandas.DataFrame
            DataFrame of unit attributes parsed from .csv
        """
        return self._unit_attrs

    @property
    def unit_info(self):
        """
        Unit info:
        ('unit_id', 'latitude', 'longitude', 'state', 'EPA_region'
         'NERC_region', 'unit_type', 'fuel_type', 'group_type')

        Returns
        ----------
        unit_info: pandas.DataFrame
            DataFrame of unit info
        """
        unit_info = self.unit_attrs[[self.COLS.keys()]]
        unit_info = unit_info.rename(columns=self.COLS)
        unit_info = self.create_unit_info(unit_info)
        return unit_info

    @staticmethod
    def _parse_csv(unit_attrs_path):
        """
        Load unit attributes from .csv file and update columns

        Parameters
        ----------
        unit_attrs_path : str
            Path to .csv containing facility (unit) attributes

        Returns
        -------
        unit_attrs : pandas.DataFrame
            DataFrame of unit attributes
        """
        unit_attrs = pd.read_csv(unit_attrs_path, index_col=False)

        return unit_attrs

    @staticmethod
    def create_group_types(unit_attrs):
        """
        Create technology (fuel) groups from fuel and unit types

        Parameters
        ----------
        unit_attrs: pandas.DataFrame
            DataFrame of unit attributes

        Returns
        -------
        group_types : pd.DataFrame
            Unit technology and fuel group type
        """
        tech = unit_attrs['unit_type'].copy()
        # Combine boilers and tangentially-fired units
        pos = tech.apply(lambda unit_type: 'boiler' in unit_type or
                         unit_type == 'Tangentially-fired')
        tech[pos] = 'Boiler'
        # Combine Combined cycle units
        pos = tech.apply(lambda unit_type: 'cycle' in unit_type)
        tech[pos] = 'CC'
        # Combine turbine units
        pos = tech.apply(lambda unit_type: 'turbine' in unit_type)
        tech[pos] = 'CT'

        fuel = unit_attrs['fuel_type'].copy()
        # Combine Natural Gas and all other gases
        pos = fuel.apply(lambda fuel_type: 'Gas' in fuel_type)
        fuel[pos] = 'NG'
        # Combine Diesel Oil and all other oils
        pos = fuel.apply(lambda fuel_type: 'Oil' in fuel_type)
        fuel[pos] = 'Oil'
        # Combine Coals and petroleum coke
        pos = fuel.apply(lambda fuel_type: 'Coal' in fuel_type or
                         fuel_type == 'Petroleum Coke')
        fuel[pos] = 'Coal'
        # Combine Wood and Other Solid Fuel and Tire Derived Fuel
        pos = fuel.apply(lambda fuel_type: fuel_type in
                         ['Wood', 'Other Solid Fuel', 'Tire Derived Fuel'])
        fuel[pos] = 'Solid Fuel'

        # Combine tech and fuel types
        group_types = tech + ' (' + fuel + ')'
        group_types.name = 'group_type'

        return group_types

    @staticmethod
    def create_unit_info(unit_attrs):
        """
        Add unit_ids and group_types to unit_attrs

        Parameters
        ----------
        unit_attrs: pandas.DataFrame
            DataFrame of unit attributes

        Returns
        -------
        unit_attrs : pd.DataFrame
            Updated and cleaned up unit attributes
        """
        unit_attrs['unit_id'] = ParseSmoke.create_unit_ids(unit_attrs)
        unit_attrs = unit_attrs.drop(columns=['ORISID', 'BLRID'])
        unit_attrs['group_type'] = ParseUnitInfo.create_group_types(unit_attrs)

        return unit_attrs

    @classmethod
    def add_unit_info(cls, smoke_df, unit_attrs_path):
        """
        Parse unit info and add it to smoke_df

        Parameters
        ----------
        smoke_df : pandas.DataFrame
            DataFrame of performance variables from SMOKE data
        unit_attrs_path : str
            Path to .csv containing facility (unit) attributes

        Returns
        -------
        smoke_df : pandas.DataFrame
            DataFrame of performance variables from SMOKE data with unit info
        """
        unit_info = cls(unit_attrs_path).unit_info
        smoke_df = pd.merge(smoke_df, unit_info, on='unit', how='left')
        smoke_df = smoke_df.fill_na('None')

        return smoke_df


class CleanSmoke:
    """
    Pre-clean Smoke data prior to Heat Rate analysis
    - Remove null value
    - Remove start-up and shut-down
    - Remove unrealistic values
    - Convert gross load to net load
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
