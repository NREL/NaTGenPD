# -*- coding: utf-8 -*-
"""
Data clustering utilities
@author: gbuster
"""

import logging
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

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


class Cluster:
    """
    DBSCAN clustering class
    """
    def __init__(self, unit_df):
        self._unit_df = unit_df
        unit_attrs = unit_df.iloc[0]
        self._unit_id = unit_attrs['unit_id']
        self._type = unit_attrs['group_type']

    def __repr__(self):
        """
        Print the Clustering class and meta shape

        Returns
        ---------
        str
            Clustering type and variables, farms in meta
        """
        name = self.__class__.__name__
        return '{} for {} unit {}'.format(name, self._type, self._unit_id)

    @staticmethod
    def normalize_values(arr, norm='max'):
        """
        Normalize values in array by column

        Parameters
        ----------
        arr : ndarray
            ndarray of values extracted from meta
            shape (n samples, with m features)
        norm : str
            Normalization method to use, see sklearn.preprocessing.normalize

        Returns
        ---------
        arr : ndarray
            array with values normalized by column
            shape (n samples, with m features)
        """
        return normalize(arr, axis=0, norm=norm)

    @staticmethod
    def n_dist(array, n):
        """
        Compute the nk-nearest neighbor distance for all points in array

        Parameters
        ----------
        array : ndarray
            Array of n samples with m features
        n : int
            Number of nearest neighbors

        Returns
        ---------
        dist : ndarray
            1d array of nk-nearest neighbor distances
        """
        def n_min(v, n):
            return np.sort(v)[1:n + 1]

        dist = squareform(pdist(array))
        dist = np.apply_along_axis(n_min, 0, dist, n).flatten()
        return np.sort(dist)

    @staticmethod
    def line_dist(k_dist):
        """
        Extract the distance between each point on the nk-nearest neighbor
        distance curve and the line connecting the min and max values. The
        maximum distance corresponds to the knee of the curve, which can be
        used to estimate the optimal epsilon value for DBSCAN

        Parameters
        ----------
        k_dist : ndarray
            1d array of nk-nearest neighbor distances

        Returns
        ---------
        dist : ndarray
            1d array of distances between each value and the line connecting
            the min and max values
        """
        coords = np.dstack((range(len(k_dist)), k_dist))[0]

        b = coords[-1] - coords[0]
        b_hat = b / np.sqrt(np.sum(b**2))
        p = coords - coords[0]

        d = p - np.outer(np.dot(p, b_hat), b_hat)
        dist = np.sqrt(np.sum(d**2, axis=1))
        return dist

    @staticmethod
    def cluster_score(arr, labels):
        """
        Modified silhouette score that excludes outliers

        Parameters
        ----------
        array : ndarray
            Array to be used for clustering, shape n samples with m features
        labels : ndarray
            Vector of cluster labels for each row in array

        Returns
        -------
        s : float
            Silhouette score computed after removing outliers (label < 0)
        """
        pos = labels >= 0
        s = silhouette_score(arr[pos], labels[pos])
        return s

    def get_data(self, cols, normalize=True, **kwargs):
        """
        Print the Clustering class and meta shape

        Parameters
        ----------
        cols : list
            List of columns to extract
        normalize : bool
            Option to normalize data or not
        kwargs : dict
            internal kwargs

        Returns
        ---------
        arr : ndarray
            Array of given column values for all farms in meta
            shape (n samples, with m features)
        """
        arr = self._unit_df[cols].values
        if normalize:
            arr = self.normalize_values(arr, **kwargs)

        return arr

    def optimal_eps(self, array, k=5):
        """
        Use the k-nearest neighbor plot to estimate the optimal epsilon

        Parameters
        ----------
        array : ndarray
            Array to be used for clustering, shape n samples with m features
        k : 'int'
            Number of nearest neighbors (corresponds to min_samples in DBSCAN)

        Returns
        ---------
        eps : float
            Optimal epsilon for running DBSCAN with min_samples = k
        """
        k_dist = self.n_dist(array, k)
        eps_dist = self.line_dist(k_dist)
        eps_pos = np.argmax(eps_dist)
        eps = k_dist[eps_pos]
        return eps

    def _DBSCAN(self, array, min_samples, eps=None):
        """
        Run DBSCAN on array, compute eps if not supplied.

        Parameters
        ----------
        array : ndarray
            Array to be used for clustering, shape n samples with m features
        min_samples : int
            min_samples value for running DBSCAN
        eps : float
            Epsilon value for running DBSCAN
            If None estimate using k-n distance and min_samples

        Returns
        ---------
        labels : ndarray
            Vector of cluster labels for each row in array
        eps : float
            eps value used to run DBSCAN
        min_samples : int
            min_samples value used to run DBSCAN
        """
        if eps is None:
            eps = self.optimal_eps(array, k=min_samples)

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(array)

        return labels, eps, min_samples

    def optimize_clusters(self, array, min_samples, dt=0.1):
        """
        Incrimentally increase eps from given value to optimize cluster
        size

        Parameters
        ----------
        array : ndarray
            Array to be used for clustering, shape n samples with m features
        min_samples : int
            min_samples value for running DBSCAN
        eps : float
            Epsilon value for running DBSCAN
            If None estimate using k-n distance and min_samples
        dt : float
            Percentage eps by which it is to be incrementally increased

        Returns
        ---------
        labels : ndarray
            Vector of cluster labels for each row in array
        eps : float
            eps value used to run DBSCAN
        min_samples : int
            min_samples value used to run DBSCAN
        """
        labels, eps, _ = self._DBSCAN(array, min_samples)
        label_n = [_l for _l in np.unique(labels) if _l != -1]
        n_clusters = len(label_n)
        eps_dt = eps * dt

        score = self.cluster_score(array, labels)
        cluster_params = labels, eps, min_samples
        while len(label_n) > 1:
            eps += eps_dt
            eps_dt = eps / dt
            labels, _, _ = self._DBSCAN(array, min_samples, eps=eps)

            label_n = [_l for _l in np.unique(labels) if _l != -1]
            if len(label_n) == n_clusters:
                s = self.cluster_score(array, labels)
                if s > score:
                    score = s
                    cluster_params = labels, eps, min_samples
            else:
                break

        labels, eps, min_samples = cluster_params
        return labels, eps, min_samples

    def get_n_clusters(self, array, n_clusters, optimize=True, min_samples=2,
                       **kwargs):
        """
        Iterate through min_samples until n_clusters are identified using
        DBSCAN

        Parameters
        ----------
        array : ndarray
            Array to be used for clustering, shape n samples with m features
        n_clusters : int
            Number of clusters to try and find by iterating through
            min_samples
        optimize : bool
            Optimize eps for n_clusters
        min_samples : int
            Starting value for min_samples
        kwargs : dict
            Internal kwargs

        Returns
        ---------
        labels : ndarray
            Vector of cluster labels for each row in array
        eps : float
            eps value used to run DBSCAN
        min_samples : int
            min_samples value used to run DBSCAN
        """
        cluster_params = None
        score = None
        while True:
            labels, eps, _ = self._DBSCAN(array, min_samples)

            label_n = [_l for _l in np.unique(labels) if _l != -1]
            if len(label_n) == n_clusters:
                if cluster_params is None:
                    cluster_params = labels, eps, min_samples
                    score = self.cluster_score(array, labels)
                else:
                    s = self.cluster_score(array, labels)
                    if s > score:
                        score = s
                        cluster_params = labels, eps, min_samples

                if optimize:
                    dt = kwargs.get('dt', 0.1)
                    cluster_params = self.optimize_clusters(array,
                                                            min_samples,
                                                            dt=dt)
            elif len(label_n) < n_clusters:
                if cluster_params is None:
                    raise RuntimeError('{:} clusters could not be found'
                                       .format(n_clusters))
                else:
                    break

            min_samples += 1

        labels, eps, min_samples = cluster_params
        return labels, eps, min_samples


class SingleCluster(Cluster):
    """
    Subclass to perform single cluster extraction on non-CC generators
    """
