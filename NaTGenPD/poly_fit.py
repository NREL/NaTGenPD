# -*- coding: utf-8 -*-
"""
Polynomial fitting utilities
@author: mrossol
"""
import logging
import numpy as np
import os
import pandas as pd
import warnings

from NaTGenPD.handler import CEMS, Fits

logger = logging.getLogger(__name__)


class PolyFit:
    """
    Fit filtered units to a polynomial
    """
    META_COLS = ['latitude', 'longitude', 'state', 'EPA_region', 'NERC_region',
                 'unit_type', 'fuel_type', 'group_type']

    def __init__(self, filtered_h5, order=4):
        """
        Parameters
        ----------
        filtered_h5 : str
            Path to .h5 file containing filtered CEMS data
        order : int
            Order of the polynomial fit
        """
        self._h5 = filtered_h5
        self._order = order

    @staticmethod
    def extract_fit(load, heat_rate, order=4, points=5):
        """
        Fit unit to a polynomial of given order

       Parameters
        ----------
        load : ndarray
            Load data for unit
        heat_rate : ndarray
            Heat Rate data for unit
        order : int
            Order/degree of the polynomial fit
        points : int
            Number of load/heat-rate points to save

        Returns
        -------
        unit_fit : pandas.Series
            Heat rate fit and meta-data for unit
            Fit parameters p = [p_n, p_n-1, ...]
            Such that y = p_n x^n + p_n-1 x^n-1 ...
        """
        fit_params = None
        load_range = None
        hr_range = None
        if load.size:
            load_min = load.min()
            load_max = load.max()
            try:
                fit = np.polyfit(load, heat_rate, order, full=True)
                if fit[2] > 1:
                    fit_params = fit[0]
                    load_range = np.linspace(load_min, load_max, points)
                    hr_range = np.poly1d(fit_params)(load_range)
            except Exception:
                logger.exception('Cannot fit unit')
        else:
            load_min = np.nan
            load_max = np.nan

        unit_fit = pd.Series(fit_params, index=['a4', 'a3', 'a2', 'a1', 'a0'])
        load_index = ['load_{:}'.format(i) for i in range(1, points + 1)]
        load_index[0] = 'load_min'
        load_index[-1] = 'load_max'

        unit_load = pd.Series(load_range, index=load_index)
        unit_load['load_min'] = load_min
        unit_load['load_max'] = load_max
        unit_load['total_load'] = load.sum()
        unit_load['min_gen_perc'] = load_min / load_max

        hr_index = ['heat_rate({:})'.format(load_i) for load_i in load_index]
        unit_hr = pd.Series(hr_range, index=hr_index)

        return pd.concat([unit_fit, unit_load, unit_hr])

    def fit_unit(self, unit_df, **kwargs):
        """
        Extract meta data and heat-rate fit(s) for given unit

        Parameters
        ----------
        unit_df : pandas.DataFrame
            DataFrame for unit to be fit
        kwargs : dict
            internal kwargs

        Returns
        -------
        unit_fit : pandas.DataFrame
            DataFrame of heat-rate fit(s) for given unit
        """
        unit_meta = unit_df.iloc[0]
        unit_id = unit_meta['unit_id']
        clusters = sorted([label for label in unit_df['cluster'].unique()
                           if label >= 0])

        if len(clusters) > 1:
            unit_fit = []
            for label in clusters:
                fit_id = '{}-{}'.format(unit_id, label)

                pos = unit_df['cluster'] == label
                cluster_df = unit_df.loc[pos, ['load', 'heat_rate']]
                cluster_df = cluster_df.sort_values(['load', 'heat_rate'])
                load = cluster_df['load'].values
                heat_rate = cluster_df['heat_rate'].values
                cluster_fit = self.extract_fit(load, heat_rate,
                                               order=self._order, **kwargs)
                cluster_fit.name = fit_id
                unit_fit.append(cluster_fit.to_frame().T)

            unit_fit = pd.concat(unit_fit)
        else:
            fit_id = '{}'.format(unit_id)
            pos = unit_df['cluster'] >= 0
            cluster_df = unit_df.loc[pos, ['load', 'heat_rate']]
            cluster_df = cluster_df.sort_values(['load', 'heat_rate'])
            load = cluster_df['load'].values
            heat_rate = cluster_df['heat_rate'].values
            cluster_fit = self.extract_fit(load, heat_rate,
                                           order=self._order, **kwargs)
            cluster_fit.name = fit_id
            unit_fit = cluster_fit.to_frame().T

        for col in self.META_COLS:
            unit_fit.loc[:, col] = unit_meta[col]

        return unit_fit

    def fit_group(self, group_type, out_file=None, **kwargs):
        """
        Extract polynomial fits for all units in given group

        Parameters
        ----------
        group_type : str
            Group type (generator type + fuel type) to filter
        out_file : str
            Path to file inwhich to save fit information (.json or .csv)
        kwargs : dict
            internal kwargs

        Returns
        -------
        group_fits : pandas.DataFrame
            DataFrame of fit information
        """
        logger.info('Fitting all {} units'.format(group_type))
        with CEMS(self._h5, mode='r') as f:
            group = f[group_type]

        group_fits = []
        for unit_id, unit_df in group.unit_dfs:
            logger.debug('- Fitting unit {}'.format(unit_id))
            group_fits.append(self.fit_unit(unit_df, **kwargs))

        group_fits = pd.concat(group_fits)
        group_fits.index.name = 'unit_id'
        if out_file:
            logger.debug('- Saving fits to {}'
                         .format(out_file))
            if out_file.endswith('.csv'):
                group_fits.to_csv(out_file)
            elif out_file.endswith('.json'):
                group_fits.to_json(out_file)
            else:
                raise ValueError('Invalid file type, cannot save to .{}'
                                 .format(os.path.splitext(out_file)[-1]))

        return group_fits

    def fit_all(self, out_dir, **kwargs):
        """
        Extract unit_fits for all units

        Parameters
        ----------
        out_dir : str
            Directory into which fit files (.csvs) should be saved
        kwargs: dict
            Internal kwargs
        """
        with CEMS(self._h5, mode='r') as f:
            group_types = f.dsets

        for g_type in group_types:
            out_path = "{}_fits.csv".format(g_type)
            out_path = os.path.join(out_dir, out_path)
            _ = self.fit_group(g_type, out_file=out_path, **kwargs)

    @classmethod
    def run(cls, filtered_h5, out_dir, order=4, **kwargs):
        """
        Extract unit_fits for all units

        Parameters
        ----------
        filtered_h5 : str
            Path to .h5 file containing filtered CEMS data
        out_dir : str
            Directory into which fit files (.csvs) should be saved
        order : int
            Order of the polynomial fit
        kwargs: dict
            Internal kwargs
        """
        fit = cls(filtered_h5, order=order)
        fit.fit_all(out_dir, **kwargs)


class GenericFit:
    """
    Create generic fits from polynomial fits
    """
    def __init__(self, fit_dir):
        """
        Parameters
        ----------
        fit_dir : str
            Path to directory containing polynomial fit .csvs
        """
        self._fit_dir = fit_dir
        self._fits = Fits(fit_dir)

    @staticmethod
    def _create_generic_fit(group_fits, method='median', points=20):
        """
        Calculate generic heat rate curve for given group type using
        given aggregation method with given number of data points

        Parameters
        ----------
        group_fits : pd.DataFrame
            DataFrame group type polynomial fits
        method : str
            Option to take the median or mean
        points : int
            Number of data points for generic fit

        Returns
        -------
        generic_curves: pd.DataFrame | NoneType
            DataFrame containing the generic heat_rate values as a function of
            normalized load
        """
        group_fits = group_fits[~group_fits['a0'].isnull()]
        if group_fits.shape[0] > 0:
            min_gen = group_fits['load_min'] / group_fits['load_max']
            if method.lower() == 'median':
                min_gen = min_gen.median()
            else:
                min_gen = min_gen.mean()

            load_norm = np.linspace(min_gen, 1, points)

            load_ranges = np.tile(load_norm, (group_fits.shape[0], 1))
            load_maxs = group_fits.as_matrix(['load_max'])
            load_ranges = load_maxs * load_ranges
            fit_params = group_fits.as_matrix(['a4', 'a3', 'a2', 'a1', 'a0'])
            poly_fits = [np.poly1d(params) for params in fit_params]

            hr_curves = np.array([poly_fit(load_range)
                                  for poly_fit, load_range
                                  in zip(poly_fits, load_ranges)])

            if method.lower() == 'median':
                hr_curve = np.median(hr_curves, axis=0)
            else:
                hr_curve = np.mean(hr_curves, axis=0)

            generic_curve = pd.DataFrame({'Normalized Load (%)': load_norm,
                                          'Heat Rate (mmBTU/MWh)': hr_curve})
        else:
            msg = '- Cannot create a generic curve, no valid units present!'
            logger.warning(msg)
            warnings.warn(msg)
            generic_curve = None

        return generic_curve

    def fit_all(self, out_dir, **kwargs):
        """
        Extract generic fits for all groups

        Parameters
        ----------
        out_dir : str
            Directory into which generic fit files (.csvs) should be saved
        kwargs: dict
            Internal kwargs
        """
        generic_fits = Fits(out_dir, suffix='generic_fit.csv')
        for g_name, group_fits in self._fits:
            logger.info('Creating Generic Fit for {}'.format(g_name))
            generic_fit = self._create_generic_fit(group_fits, **kwargs)
            if generic_fit is not None:
                generic_fits[g_name] = generic_fit

    @classmethod
    def run(cls, fit_dir, out_dir, method='median', points=20):
        """
        Create generic fits for all group types

        Parameters
        ----------
        fit_dir : str
            Path to directory containing polynomial fit .csvs
        out_dir : str
            Directory into which generic fit files (.csvs) should be saved
        method : str
            Option to take the median or mean
        points : int
            Number of data points for generic fit
        """
        generic = cls(fit_dir)
        generic.fit_all(out_dir, method=method, points=points)
