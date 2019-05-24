# -*- coding: utf-8 -*-
"""
Data filtering utilities
@author: mrossol
"""
import concurrent.futures as cf
import logging
import numpy as np
import os
import pandas as pd
from .cluster import SingleCluster, ClusterCC
from .handler import CEMS

logger = logging.getLogger(__name__)


class Filter:
    """
    Run Cluster filters on all units
    """
    FILTERS = {'Boiler': SingleCluster,
               'CT': SingleCluster,
               'CC': ClusterCC}

    def __init__(self, clean_h5, years=1):
        """
        Parameters
        ----------
        clean_h5 : str
            Path to .h5 file with pre-cleaned CEMS data
        years : int
            Number of years of data being filtered
        """
        self._clean_h5 = clean_h5
        self._years = years

    @property
    def total_points(self):
        """
        Number of total points possible in dataset

        Returns
        -------
        points : int
            total number of possible timesteps
        """
        points = self._years * 8760
        return points

    def filter_group(self, group_type, workers=None, **kwargs):
        """
        Filter all units of given group_type

        Parameters
        ----------
        group_type : str
            Group type (generator type + fuel type) to filter
        workers : int | NoneType
            If not None, number of parallel workers to use

        Returns
        -------
        group_df : pd.DataFrame
            Updated group DataFrame with cluster labels post filtering
        """
        logger.info('Filtering all {} units'.format(group_type))
        total_points = self.total_points
        min_samples = int(total_points / 1000)
        logger.debug('\t- Using min_samples = {}'.format(min_samples))
        threshold = int(total_points / 100)
        logger.debug('\t- Skipping units with < {} points'.format(threshold))
        filter = self.FILTERS[group_type.split(' (')[0]]

        with CEMS(self._clean_h5, mode='r') as f:
            group = f[group_type]

        if workers:
            with cf.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = []
                for unit_id, unit_df in group.unit_dfs:
                    logger.debug('- Filtering unit {}'.format(unit_id))
                    futures.append(executor.submit(filter, unit_df,
                                                   min_samples,
                                                   threshold=threshold,
                                                   **kwargs))

                group_df = [future.result() for future in futures]
        else:
            group_df = []
            for unit_id, unit_df in group.unit_dfs:
                logger.debug('- Filtering unit {}'.format(unit_id))
                group_df.append(filter(unit_df, min_samples,
                                       threshold=threshold, **kwargs))

        return pd.concat(group_df)

    def filter_all(self, out_h5, **kwargs):
        """
        Filter all groups in clean_h5 and save to out_h5

        Parameters
        ----------
        out_h5 : str
            Path to .h5 file into which filtered data should be saved
        kwargs : dict
            Internal kwargs
        """
        with CEMS(self._clean_h5, mode='r') as f_in:
            group_types = f_in.dsets

        with CEMS(out_h5, mode='w') as f_out:
            for g_type in group_types:
                f_out[g_type] = self.filter_group(g_type, **kwargs)

    @classmethod
    def run(cls, clean_h5, out_h5, years=1, **kwargs):
        """
        Filter all groups in clean_h5 and save to out_h5

        Parameters
        ----------
        clean_h5 : str
            Path to .h5 file with pre-cleaned CEMS data
        out_h5 : str
            Path to .h5 file into which filtered data should be saved
        years : int
            Number of years worth of data being filtered
        kwargs : dict
            Internal kwargs
        """
        f = cls(clean_h5, years=years)
        f.filter_all(out_h5, **kwargs)


class PolyFit:
    """
    Fit filtered units to a polynomial
    """
    META_COLS = ['latitude', 'longitude', 'state', 'EPA_region', 'NERC_region',
                 'unit_type', 'fuel_type', 'group_type', 'cts']

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
                if fit[2] <= 1:
                    raise RuntimeError('Final fit is poorly conditioned!')

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
        clusters = [label for label in unit_df['cluster'].unique()
                    if label >= 0]

        unit_fit = []
        if len(clusters) > 1:
            cluster_id = True

        for label in clusters:
            id = '{}'.format(unit_id)
            if cluster_id:
                id += '-{}'.format(label)

            pos = unit_df['cluster'] == label
            cluster_df = unit_df.loc[pos, ['load', 'heat_rate']]
            cluster_df = cluster_df.sort_values(['load', 'heat_rate'])
            load = cluster_df['load'].values
            heat_rate = cluster_df['heat_rate'].values
            cluster_fit = self.extract_fit(load, heat_rate,
                                           order=self._order, **kwargs)
            cluster_fit.name = id
            unit_fit.append(cluster_fit.to_frame().T)

        unit_fit = pd.concat(unit_fit)
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
        logger.info('Filtering all {} units'.format(group_type))
        with CEMS(self._h5, mode='r') as f:
            group = f[group_type]

        group_fits = []
        for unit_id, unit_df in group.unit_dfs:
            logger.debug('- Filtering unit {}'.format(unit_id))
            group_fits.append(self.fit_unit(unit_df, **kwargs))

        group_fits = pd.concat(group_fits)
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


def get_hr_min(unit, points=100):
    """
    Extract minimum heat rate from unit fits

    Parameters
    ----------
    unit : pandas.Series
        Row containing the fit stats and parameters for a single unit

    Returns
    -------
    hr_min: float
        Minimum heat rate from heat rate curve fit
    """
    if not unit['a4'].isnull():
        fit_params = unit[['a4', 'a3', 'a2', 'a1', 'a0']].values
        poly_fit = np.poly1d(fit_params)
        load_min, load_max = unit[['load_min', 'load_max']].values
        x = np.linspace(load_min, load_max, points)
        hr_min = poly_fit(x).min()
    else:
        hr_min = None

    return hr_min


def min_hr_filter(group_fits, stdev_multiplier=2, threshold=(None, None)):
    """
    Filter out the most and least efficient units based on multiples of the
    standard deviation from the mean

    Parameters
    ----------
    group_fits : pandas.DataFrame
        DataFrame of fits to filter
    stdev_multiplier : float
        Multiple of the stdev from the mean to use as the filter thresholds
    threshold : tuple
        Threshold(s) to use instead of the above multiplier

    Returns
    -------
    group_fits : pandas.DataFrame
        Filtered group fits
    """
    min_hr = group_fits.apply(get_hr_min, axis=1).dropna()
    mean = min_hr.mean()
    stdev = min_hr.stdev()
    thresh = np.array([-stdev_multiplier, stdev_multiplier]) * stdev + mean
    for i, t in enumerate(threshold):
        if t is not None:
            thresh[i] = t

    pos = np.logical_or(min_hr < thresh[0], min_hr > thresh[1])
    idx = min_hr[~pos].index
    null_cols = [c for c in group_fits.columns
                 if c.startswith(('a', 'heat_rate'))]
    group_fits.loc[idx, null_cols] = None

    return group_fits
