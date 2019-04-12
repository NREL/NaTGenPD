# -*- coding: utf-8 -*-
"""
Data clustering utilities
@author: gbuster
"""

import logging
import numpy as np
from scipy.spatial import cKDTree


logger = logging.getLogger(__name__)


def single_cluster(df, cols=None, NN=5, dist=0.1, normalize=True):
    """
    Use euclidian KNN to reduce noisy dataset to one cluster.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to cluster.
    cols : list | NoneType
        Columns in df to cluster on. None clusters on all columns in df.
    NN : int
        Number of desired nearest neighbors to be considered in the central
        cluster.
    dist : int or float
        Euclidian distance between coordinates to be considered a neighbor.
    normalize : bool
        Option to perform range-normalization on data before clustering.

    Returns
    -------
    df : pd.DataFrame
        Subset of input df with outlier data removed.
    """

    n_dat = len(df)
    if n_dat <= 1:
        # coordinate set with only one coordinate cannot be cleaned further
        return df

    NN = np.min([NN, int(n_dat / 3) + 1])
    N = int(np.ceil(NN)) + 2
    if N > n_dat:
        raise ValueError('Number of nearest neighbors to retrieve (N={}) '
                         'exceeds length of the dataframe {}'.format(N, n_dat))
    logger.debug('Set contains {} coordinates, N set to {}, '
                 'N-neighbors required: {} at {}'
                 .format(n_dat, N, NN, dist))

    # get nearest neighbors
    d, _ = knn(df, cols, return_dist=True, normalize=normalize, k=N)

    # count the number of distance results per index less than the threshold
    # (exclude the 1st dist row which is self-referential)
    counts = np.sum(d[:, 1:] < dist, axis=1)

    # find where there are NN neighbors satisfying the distance threshold
    mask = np.where(counts > NN)[0]

    logger.debug('{} points after cleaning'.format(len(mask)))

    if len(mask) > 1:
        # multiple cleaned points exist, return them
        return df.iloc[mask, :]
    else:
        # Cleanup eliminated all points. Retry with less strict thresholds
        return single_cluster(df, cols=cols, NN=int(NN * 0.8), dist=dist * 1.2,
                              normalize=normalize)


def knn(df, cols, return_dist=False, normalize=True, k=1):
    """Get euclidian distance nearest neighbors for numerical column data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with data to cluster.
    cols : list | NoneType
        Columns in df to cluster on. None clusters on all columns in df.
    return_dist : bool
        Determines whether this method returns (dist, ind) or just ind
    normalize : str
        Determines whether the data is range-normalized by column.
    k : int
        The number of k-th nearest neighbors to return.

    Returns
    -------
    dist : np.ndarray
        Each entry gives the list of distances to the neighbors of the
        corresponding point.
    ind : np.ndarray
        Each entry gives the list of indices of neighbors of the corresponding
        point.
    """

    # take slices of df1/df2 based on col slices
    if cols is None:
        cols = df.columns.values

    array1 = df.loc[:, cols].values
    array2 = df.loc[:, cols].values

    try:
        # make sure these columns can be cast as float (required for KNN)
        array1 = array1.astype(dtype=float)
        array2 = array2.astype(dtype=float)
    except Exception as e:
        raise e

    if normalize:
        for i, _ in enumerate(cols):
            # get min and max values from a concatenation of both arrays
            min_all = np.nanmin((np.nanmin(array1[:, i]),
                                 np.nanmin(array2[:, i])))
            max_all = np.nanmax((np.nanmax(array1[:, i]),
                                 np.nanmax(array2[:, i])))
            range_all = max_all - min_all

            if range_all == 0.0:
                # protect against div by zero
                range_all = 1

            # range scale from 0 to 1
            array1[:, i] -= min_all
            array1[:, i] /= range_all
            array2[:, i] -= min_all
            array2[:, i] /= range_all

    # execute KNN query
    tree = cKDTree(array2)
    dist_1, ind_1 = tree.query(array1, k=k)

    logger.debug('KNN index nearest neighbors: \n{}'.format(ind_1))
    logger.debug('KNN distance: \n{}'.format(dist_1))

    if return_dist:
        return dist_1, ind_1
    else:
        return ind_1
