# -*- coding: utf-8 -*-
"""
Data clustering utilities

@author: mrossol
"""
from functools import partial
import numpy as np
import os
import pandas as pd
import warnings


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
        unit_info = self.unit_attrs[list(self.COLS.keys())]
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
        # Combine Coals and petroleum coke
        pos = fuel.apply(lambda fuel_type: 'Coal' in fuel_type or
                         fuel_type == 'Petroleum Coke')
        fuel[pos] = 'Coal'
        # Combine Natural Gas and all other gases
        pos = fuel.apply(lambda fuel_type: 'Gas' in fuel_type)
        fuel[pos] = 'NG'
        # Combine Diesel Oil and all other oils
        pos = fuel.apply(lambda fuel_type: 'Oil' in fuel_type)
        fuel[pos] = 'Oil'
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
    - Convert gross load to net load
    - Remove null value
    - Remove start-up and shut-down
    - Remove unrealistic values
    """
    def __init__(self, smoke, unit_attrs_path=None):
        """
        Parameters
        ----------
        smoke : pandas.DataFrame | str
            DataFrame of performance variables or path to .h5 file
        unit_attrs_path : str
            Path to .csv containing facility (unit) attributes
        """
        self._smoke_df = self.load_smoke_df(smoke,
                                            unit_attrs_path=unit_attrs_path)

    @property
    def smoke_df(self):
        """
        DataFrame of performance variables from SMOKE data with unit info

        Returns
        -------
        _smoke_df : pandas.DataFrame
            DataFrame of performance variables from SMOKE data with unit info
        """
        return self._smoke_df

    @staticmethod
    def load_smoke_df(smoke_df, unit_attrs_path=None):
        """
        Load smoke data if needed and combine unit info if needed

        Parameters
        ----------
        smoke_df : pandas.DataFrame | str
            DataFrame of performance variables or path to .h5 file
        unit_attrs_path : str
            Path to .csv containing facility (unit) attributes

        Returns
        -------
        smoke_df : pandas.DataFrame
            DataFrame of performance variables from SMOKE data with unit info
        """
        if isinstance(smoke_df, str):
            smoke_df = pd.read_hdf(smoke_df, 'smoke_df')

        if 'group_type' not in smoke_df.columns:
            if unit_attrs_path is None:
                raise ValueError('Unit attributes are needed to clean data')
            else:
                smoke_df = ParseUnitInfo.add_unit_info(smoke_df,
                                                       unit_attrs_path)

        return smoke_df

    @staticmethod
    def gross_to_net(smoke_df,
                     load_multipliers={'solid': 0.925, 'liquid': 0.963}):
        """
        Convert gross load to net load using given multipliers for
        solid and liquid fuel

        Parameters
        ----------
        smoke_df : pandas.DataFrame
            DataFrame of performance variables from SMOKE data with unit info
        load_multipliers : dict
            Gross to net multipliers for solid and liquid/gas fuel

        Returns
        -------
        smoke_df : pandas.DataFrame
            Updated DataFrame w/ net load and updated heat rate
        """
        fuel_map = {'solid': ['Coal', 'Solid Fuel'], 'liquid': ['NG', 'Oil']}
        smoke_df = smoke_df.rename(columns={'gload': 'load'})

        def _filter(fuels, row):
            for x in fuels:
                if x in row:
                    return True
            return False

        for key, fuels in fuel_map.items():
            pos = smoke_df['group_type'].apply(partial(_filter, fuels))
            gross_to_net = load_multipliers[key]
            smoke_df.loc[pos, 'load'] *= gross_to_net
            smoke_df.loc[pos, 'heat_rate'] *= (1 / gross_to_net)

        return smoke_df

    @staticmethod
    def remove_null_values(smoke_df, hr_bounds=(4.5, 40), **kwargs):
        """
        Remove null values:
        - HTINPUT <= 0
        - HTINPUTMEASURE is not measured (1) or calculated (2)
        - load <= 0
        - heat_rate is outside given heat reat bounds

        Parameters
        ----------
        smoke_df : pandas.DataFrame
            DataFrame of performance variables from SMOKE data with unit info
        hr_bounds : tuple
            Bounds (min, max) of realistic heat_rate values
        kwargs : dict
            Internal kwargs

        Returns
        -------
        smoke_df : pandas.DataFrame
            Updated DataFrame w/ null values removed
        """
        smoke_df = smoke_df.loc[smoke_df['HTINPUT'] > 0]
        smoke_df = smoke_df.loc[smoke_df['HTINPUTMEASURE'].isin([1, 2])]

        if 'load' not in smoke_df.columns:
            warnings.warn('Converting gload to net load')
            smoke_df = CleanSmoke.gross_to_net(smoke_df, **kwargs)

        smoke_df = smoke_df.loc[smoke_df['load'] > 0]

        hr_min = smoke_df['heat_rate'] > hr_bounds[0]
        hr_max = smoke_df['heat_rate'] < hr_bounds[1]
        smoke_df = smoke_df.loc[np.logical_and(hr_min, hr_max)]

        return smoke_df.reset_index(drop=True)

    @staticmethod
    def remove_start_stop(smoke_df, max_perc=0.1):
        """
        Remove data associated with start-up and shut-down:
        - OPTIME < 1
        - Data with < max_perc of max load or max HTINPUT

        Parameters
        ----------
        smoke_df : pandas.DataFrame
            DataFrame of performance variables from SMOKE data with unit info
        max_perc : float
            Percentage (as a float) of max load and max HTINPUT to associate
            with start-up and shut-down

        Returns
        -------
        smoke_df : pandas.DataFrame
            Updated DataFrame start-up and shut-down removed
        """
        smoke_df = smoke_df.loc[smoke_df['OPTIME'] == 1]

        if max_perc:
            maxes = smoke_df.groupby('unit_id')[['load', 'HTINPUT']].max()
            maxes = pd.merge(smoke_df['unit_id'], maxes.reset_index(),
                             on='unit_id')

            load_pos = (smoke_df['load'] / maxes['load']) > max_perc
            ht_pos = (smoke_df['HTINPUT'] / maxes['HTINPUT']) > max_perc
            smoke_df = smoke_df.loc[np.logical_and(load_pos, ht_pos)]

        return smoke_df.reset_index(drop=True)

    def preclean(self, load_multipliers={'solid': 0.925, 'liquid': 0.963},
                 hr_bounds=(4.5, 40), max_perc=0.1):
        """
        Clean-up SMOKE data for heat rate analysis:
        - Convert gross load to net load
        - Remove null/unrealistic values
        - Remove start-up and shut-down

        Parameters
        ----------
        load_multipliers : dict
            Gross to net multipliers for solid and liquid/gas fuel
        hr_bounds : tuple
            Bounds (min, max) of realistic heat_rate values
        max_perc : float
            Percentage (as a float) of max load and max HTINPUT to associate
            with start-up and shut-down

        Returns
        -------
        smoke_clean : pandas.DataFrame
            Cleaned SMOKE data
        """
        smoke_clean = self.gross_to_net(self.smoke_df,
                                        load_multipliers=load_multipliers)
        smoke_clean = self.remove_null_values(smoke_clean, hr_bounds=hr_bounds)
        smoke_clean = self.remove_start_stop(smoke_clean, max_perc=max_perc)

        return smoke_clean

    @classmethod
    def clean(cls, smoke, unit_attrs_path=None,
              load_multipliers={'solid': 0.925, 'liquid': 0.963},
              hr_bounds=(4.5, 40), max_perc=0.1):
        """
        Clean-up SMOKE data for heat rate analysis:
        - Convert gross load to net load
        - Remove null/unrealistic values
        - Remove start-up and shut-down

        Parameters
        ----------
        smoke : pandas.DataFrame | str
            DataFrame of performance variables or path to .h5 file
        unit_attrs_path : str
            Path to .csv containing facility (unit) attributes
        load_multipliers : dict
            Gross to net multipliers for solid and liquid/gas fuel
        hr_bounds : tuple
            Bounds (min, max) of realistic heat_rate values
        max_perc : float
            Percentage (as a float) of max load and max HTINPUT to associate
            with start-up and shut-down

        Returns
        -------
        smoke_clean : pandas.DataFrame
            Cleaned SMOKE data
        """
        smoke = cls(smoke, unit_attrs_path=unit_attrs_path)
        smoke_clean = smoke.preclean(load_multipliers=load_multipliers,
                                     hr_bounds=hr_bounds, max_perc=max_perc)
        return smoke_clean
