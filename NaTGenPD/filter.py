# -*- coding: utf-8 -*-
"""
Data filtering utilities
@author: mrossol
"""
import concurrent.futures as cf
import logging
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
            Group type to filter
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
