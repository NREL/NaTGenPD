# -*- coding: utf-8 -*-
"""
Heat Rate Analysis utilities
@author: mrossol
"""
import logging
import numpy as np
import os
import pandas as pd

from NaTGenPD.handler import Fits, CEMS, CEMSGroup

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)


class ProcedureAnalysis:
    """
    Analyze cleaning and filtering procedure
    """
    def __init__(self, hr_fits, raw_cems, cleaned_cems, filtered_cems,
                 cc_map_path=None):
        """
        Parameters
        ----------
        hr_fits : str
            Path to heat rate fit .csv(s)
        raw_cems : str
            Path to raw CEMS .h5 file(s)
        cleaned_cems : str
            Path to cleaned CEMS .h5 file
        filtered_cems : str
            Path to filtered CEMS .h5 file
        cc_map_path : str
            Path to cc_mapping
        """
        self._fits = Fits(hr_fits)
        self._raw_df = self.parse_raw_CEMS(raw_cems)
        self._cleaned_path = cleaned_cems
        self._filtered_path = filtered_cems
        if cc_map_path is None:
            cc_map_path = os.path.join(os.path.dirname(PROJECT_ROOT), 'bin',
                                       'cems_cc_mapping.csv')

        self._cc_map = self.load_cc_map(cc_map_path)

    @staticmethod
    def parse_raw_CEMS(raw_cems):
        """
        Combine multiple years of raw CEMS data

        Parameters
        ----------
        raw_cems : list | str
            Path to single or multiple .h5 files containing raw CEMS data

        Returns
        -------
        raw_df : pd.DataFrame
            DataFrame of raw CEMS data from all input years
        """
        if not isinstance(raw_cems, list):
            raw_cems = [raw_cems]

        raw_df = []
        for raw_file in raw_cems:
            logger.debug('\t- Loading {}'.format(os.path.basename(raw_file)))
            with CEMS(raw_file, 'r') as f:
                raw_df.append(f['raw_CEMS'].df)

        return pd.concat(raw_df)

    @staticmethod
    def load_cc_map(cc_map_path=None):
        """
        Load cc_mapping, if path is None use provided mapping in bin

        Parameters
        ----------
        cc_map_path : str
            Path to .csv with CEMS to EIA CC unit mapping

        Returns
        -------
        cc_map : pd.DataFrame
            Mapping of CEMS cc cferators (CTs) to final CC units
        """
        if cc_map_path is None:
            cc_map_path = os.path.join(os.path.dirname(PROJECT_ROOT), 'bin',
                                       'cems_cc_mapping.csv')

        cc_map = pd.read_csv(cc_map_path)
        cc_map = cc_map.rename(columns={'CCUnit': 'cc_unit',
                                        'CEMSUnit': 'unit_id'})
        cc_map = cc_map[['unit_id', 'cc_unit']]
        return cc_map

    def _get_fits(self, group_type):
        """
        Extract desired group type from heat rate fits

        Parameters
        ----------
        group_type : str
            Fuel/cferator type of interest

        Returns
        -------
        group_fits : pd.DataFrame
        """
        group_fits = self._fits[group_type]

        return group_fits

    def _get_raw(self, group_type):
        """
        Extract desired group type from raw CEMS data

        Parameters
        ----------
        group_type : str
            Fuel/cferator type of interest

        Returns
        -------
        raw : CEMSGroup
        """
        group_fits = self._get_fits(group_type)
        units = list(group_fits['unit_id'].values)
        if "CC" in group_type:
            pos = self._cc_map['cc_unit'].isin(units)
            units = self._cc_map.loc[pos, 'unit_id'].to_list()

        pos = self._raw_df['unit_id'].isin(units)
        raw = CEMSGroup(self._raw_df.loc[pos])

        return raw

    def _get_cleaned(self, group_type):
        """
        Extract desired group type from cleaned CEMS data

        Parameters
        ----------
        group_type : str
            Fuel/cferator type of interest

        Returns
        -------
        cleaned : CEMSGroup
        """
        with CEMS(self._cleaned_path, mode='r') as f:
            cleaned = f[group_type]

        return cleaned

    def _get_filtered(self, group_type):
        """
        Extract desired group type from filtered CEMS data

        Parameters
        ----------
        group_type : str
            Fuel/cferator type of interest

        Returns
        -------
        filtered : CEMSGroup
        """
        with CEMS(self._filtered_path, mode='r') as f:
            filtered = f[group_type]

        return filtered

    @staticmethod
    def _raw_stats(raw_df, unit_id, group_stats, unit_stats):
        """
        Compute raw stats for desired unit

        Parameters
        ----------
        raw_df : CEMSGroup
            Instance of CEMSGroup containing raw CEMS data
        unit_id : str
            Unit id of interest
        group_stats : pd.Series
            Aggregate stats
        unit_stats : pd.Series
            Stats for individual units

        Returns
        -------
        group_stats : pd.Series
            Updated aggregate stats
        unit_stats : pd.Series
            Updated stats for individual units
        """
        try:
            logger.debug('\t-- Extracting raw stats')
            unit_df = raw_df[unit_id]
            group_stats['unit_dfs'] += 1
            cf = unit_df['gload'].max()
            group_stats['raw_cf'] += cf
            unit_stats['raw_cf'] = cf
            points = len(unit_df)
            group_stats['total_points'] += points
            unit_stats['total_points'] = points
            non_zero = (unit_df['gload'] > 0).sum()
            group_stats['non_zero_points'] += non_zero
            unit_stats['non_zero_points'] = non_zero
        except KeyError:
            logger.debug('- {} is not present in Raw CEMS data'
                         .format(unit_id))

        return group_stats, unit_stats

    @staticmethod
    def _clean_stats(clean_df, unit_id, group_stats, unit_stats):
        """
        Compute cleaning stats for desired unit

        Parameters
        ----------
        clean_df : CEMSGroup
            Instance of CEMSGroup containing cleaned CEMS data
        unit_id : str
            Unit id of interest
        group_stats : pd.Series
            Aggregate stats
        unit_stats : pd.Series
            Stats for individual units

        Returns
        -------
        group_stats : pd.Series
            Updated aggregate stats
        unit_stats : pd.Series
            Updated stats for individual units
        """
        try:
            unit_df = clean_df[unit_id]
            if unit_df['load'].nonzero()[0].any():
                logger.debug('\t-- Extracting clean stats')
                group_stats['clean_units'] += 1
                cf = np.nanmax(unit_df['load'])
                group_stats['clean_cf'] += cf
                unit_stats['clean_cf'] = cf
        except KeyError:
            logger.debug('- {} is not present in Clean CEMS data'
                         .format(unit_id))

        return group_stats, unit_stats

    @staticmethod
    def _filter_stats(filtered_df, unit_id, unit_fit, group_stats, unit_stats):
        """
        Compute filtering stats for desired unit

        Parameters
        ----------
        filtered_df : CEMSGroup
            Instance of CEMSGroup containing filtered CEMS data
        unit_id : str
            Unit id of interest
        unit_fit : pd.Series
            Unit fit data
        group_stats : pd.Series
            Aggregate stats
        unit_stats : pd.Series
            Stats for individual units

        Returns
        -------
        group_stats : pd.Series
            Updated aggregate stats
        unit_stats : pd.Series
            Updated stats for individual units
        """
        try:
            unit_df = filtered_df[unit_id]
            pos = unit_df['cluster'] >= 0
            unit_df = unit_df.loc[pos]
            if unit_df['load'].nonzero()[0].any():
                logger.debug('\t-- Extracting filter stats')
                group_stats['filtered_units'] += 1
                cf = np.nanmax(unit_df['load'])
                group_stats['filtered_cf'] += cf
                unit_stats['filtered_cf'] = cf
                if not np.isnan(unit_fit['a0']):
                    logger.debug('\t-- Extracting final stats')
                    group_stats['final_units'] += 1
                    group_stats['final_cf'] += cf
                    unit_stats['final_cf'] = cf
                    f_points = len(unit_df.loc[unit_df['load'] > 0])
                    unit_stats['final_points'] = f_points
                    group_stats['final_points'] += f_points
        except KeyError:
            logger.debug('- {} is not present in Filtered CEMS data'
                         .format(unit_id))

        return group_stats, unit_stats

    def _group_stats(self, group_type):
        """
        Compute process stats for group in aggregate and each unit in
        group type

        Parameters
        ----------
        group_type : str
            Group (fuel/cferator) type to analyze

        Returns
        -------
        group_stats : pd.Series
            Aggregated processing stats
        group_unit_stats : pd.DataFrame
            Processing stats for each unit
        """
        group_stats = pd.Series(0, index=['unit_dfs', 'raw_cf',
                                          'total_points', 'non_zero_points',
                                          'clean_units', 'clean_cf',
                                          'filtered_units', 'filtered_cf',
                                          'final_units', 'final_cf',
                                          'final_points'])
        stats = group_stats.copy().drop(labels=['unit_dfs', 'clean_units',
                                                'filtered_units',
                                                'final_units'])
        group_stats.name = group_type

        group_fits = self._get_fits(group_type).set_index('unit_id')
        raw_df = self._get_raw(group_type)
        clean_df = self._get_cleaned(group_type)
        filtered_df = self._get_filtered(group_type)

        group_unit_stats = []
        for unit_id, unit_fit in group_fits.iterrows():
            logger.debug('\t- Extracting stats for unit: {}'.format(unit_id))
            unit_stats = stats.copy()
            unit_stats.name = unit_id
            # Raw Dat Stats
            group_stats, unit_stats = self._raw_stats(raw_df, unit_id,
                                                      group_stats, unit_stats)
            # Clean Data Stats

            group_stats, unit_stats = self._clean_stats(clean_df, unit_id,
                                                        group_stats,
                                                        unit_stats)
            # Filtered and Final data Stats
            group_stats, unit_stats = self._filter_stats(filtered_df, unit_id,
                                                         unit_fit,
                                                         group_stats,
                                                         unit_stats)
            group_unit_stats.append(unit_stats)

        group_unit_stats = pd.concat(group_unit_stats, axis=1).T

        return group_stats, group_unit_stats

    def process_stats(self, out_file):
        """
        Compute process stats for all available group types in CEMS data

        Parameters
        ----------
        out_file : str
            Path to output file to save stats to
        """
        process_stats = []
        for g_type in self._fits.group_types:
            logger.info('Extracting stats for {}'.format(g_type))
            group_stats, group_unit_stats = self._group_stats(g_type)
            process_stats.append(group_stats)

            f_name = os.path.basename(out_file)
            units_file = "{}_{}".format(g_type, f_name)
            units_file = out_file.replace(f_name, units_file)
            group_unit_stats.to_csv(units_file)

        process_stats = pd.concat(process_stats, axis=1).T
        process_stats.to_csv(out_file)

    @classmethod
    def stats(cls, hr_fits, raw_cems, cleaned_cems, filtered_cems, out_file,
              cc_map_path=None):
        """
        Compute process stats for all available group types in CEMS data

        Parameters
        ----------
        hr_fits : str
            Path to heat rate fit .csv(s)
        raw_cems : str
            Path to raw CEMS .h5 file(s)
        cleaned_cems : str
            Path to cleaned CEMS .h5 file
        filtered_cems : str
            Path to filtered CEMS .h5 file
        out_file : str
            Path to output file to save stats to
        cc_map_path : str
            Path to cc_mapping
        """
        analysis = cls(hr_fits, raw_cems, cleaned_cems, filtered_cems,
                       cc_map_path=cc_map_path)
        analysis.process_stats(out_file)


class QuartileAnalysis:
    """
    Analyze cferation and operation time by load quartile
    """
    def __init__(self, hr_fits, filtered_cems):
        """
        Parameters
        ----------
        hr_fits : str
            Path to heat rate fit .csv(s)
        filtered_cems : str
            Path to filtered CEMS .h5 file
        """
        self._fits = Fits(hr_fits)
        self._filtered_path = filtered_cems

    def __getitem__(self, group_type):
        """
        Extract desired group type from filtered CEMS data

        Parameters
        ----------
        group_type : str
            Fuel/cferator type of interest

        Returns
        -------
        group_filtered : CEMSGroup
            Filtered units for desired group with proper final heat-rate fits
        """
        group_fits = self._fits[group_type]
        pos = group_fits['a0'].isnull()
        group_fits = group_fits.loc[~pos]
        cols = [c for c in group_fits.columns if 'heat_rate' in c]
        group_fits['ave_heat_rate'] = group_fits[cols].mean()
        group_fits = group_fits[['unit_id', 'load_max', 'ave_heat_rate']]

        with CEMS(self._filtered_path, mode='r') as f:
            group_filtered = f[group_type].df

        pos = group_filtered['cluster'] >= 0
        group_filtered = group_filtered.loc[pos, ['unit_id', 'load']]

        group_filtered = pd.merge(group_filtered, group_fits,
                                  on='unit_id', how='left')
        group_filtered['cf'] = (group_filtered['load']
                                / group_filtered['load_max'])

        return group_filtered

    @staticmethod
    def _compute_stats(filtered_df):
        """
        Extract CF quartile stats for given group:
        - fraction of generation in each quartile
        - fraction of time in each quartile

        Parameters
        ----------
        filtered_df : pd.DataFrame
            DataFrame to compute stats from

        Returns
        -------
        quartile_stats : pd.Series
            CF quartile stats
        """
        quartile_stats = pd.Series()
        load = filtered_df['cf']
        load_range = (load.max() - load.min())
        bin_size = load_range / 4
        bins = load.min() + np.arange(1, 4) * bin_size
        total_cf = filtered_df['cf'].sum()
        total_points = len(filtered_df)
        # Q1
        pos = load <= bins[0]
        q_cf = filtered_df.loc[pos, 'cf'].sum()
        quartile_stats['Q1_gen_frac'] = q_cf / total_cf
        quartile_stats['Q1_time_frac'] = pos.sum() / total_points
        # Q2
        pos = (load > bins[0]) & (load <= bins[1])
        q_cf = filtered_df.loc[pos, 'cf'].sum()
        quartile_stats['Q2_gen_frac'] = q_cf / total_cf
        quartile_stats['Q2_time_frac'] = pos.sum() / total_points
        # Q3
        pos = (load > bins[1]) & (load <= bins[2])
        q_cf = filtered_df.loc[pos, 'cf'].sum()
        quartile_stats['Q3_gen_frac'] = q_cf / total_cf
        quartile_stats['Q3_time_frac'] = pos.sum() / total_points
        # Q4
        pos = load > bins[2]
        q_cf = filtered_df.loc[pos, 'cf'].sum()
        quartile_stats['Q4_gen_frac'] = q_cf / total_cf
        quartile_stats['Q4_time_frac'] = pos.sum() / total_points

        return quartile_stats

    @staticmethod
    def _hr_stats(filtered_df, bins=3):
        """
        Compute quartile stats for bins of average heat-rate

        Parameters
        ----------
        filtered_df : pd.DataFrame
            DataFrame to compute stats from

        Returns
        -------
        hr_stats : pd.Series
            CF quartile stats by average heat-rate bin
        """
        hr_stats = []
        _, bins = np.histogram(filtered_df['ave_heat_rate'].values, bins=bins)
        for i in range(bins):
            s, e = bins[[i, i + 1]]
            pos = (filtered_df['ave_heat_rate'] > s
                   & filtered_df['ave_heat_rate'] <= e)
            df = filtered_df.loc[pos]
            bin_stats = QuartileAnalysis._compute_stats(df)
            bin_stats.name = "bin_{}".format(i)
            hr_stats.append(bin_stats)

        return pd.concat(hr_stats, axis=1).T

    def quartile_stats(self, out_file, **kwargs):
        """
        Compute process stats for all available group types in CEMS data

        Parameters
        ----------
        out_file : str
            Path to output file to save stats to
        kwargs : dict
            Internal kwargs
        """
        quartile_stats = []
        for g_type in self._fits.group_types:
            logger.info('Extracting stats for {}'.format(g_type))
            group_filtered = self[g_type]
            group_stats = self._compute_stats(group_filtered)
            group_stats.name = g_type
            quartile_stats.append(group_stats)

            f_name = os.path.basename(out_file)
            group_file = "{}_{}".format(g_type, f_name)
            group_file = out_file.replace(f_name, group_file)
            hr_stats = self._hr_stats(group_filtered, **kwargs)
            hr_stats.to_csv(group_file)

        quartile_stats = pd.concat(quartile_stats, axis=1).T
        quartile_stats.to_csv(out_file)

    @classmethod
    def stats(cls, hr_fits, filtered_cems, out_file, **kwargs):
        """
        Compute quartile stats for all available group types in CEMS data

        Parameters
        ----------
        hr_fits : str
            Path to heat rate fit .csv(s)
        filtered_cems : str
            Path to filtered CEMS .h5 file
        out_file : str
            Path to output file to save stats to
        kwargs : dict
            Internal kwargs
        """
        analysis = cls(hr_fits, filtered_cems)
        analysis.quartile_stats(out_file, **kwargs)
