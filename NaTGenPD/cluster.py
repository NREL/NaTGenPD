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


class Cluster:
    """
    DBSCAN clustering class
    """
    def __init__(self, unit_df):
        """
        Parameters
        ----------
        unit_df : pandas.DataFrame
            DataFrame of timeseries heat rate data for unit of interest
        """
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
    def normalize_values(arr, norm=None):
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
        if norm:
            arr = normalize(arr, axis=0, norm=norm)
        else:
            min_all = arr.min(axis=0)
            max_all = arr.max(axis=0)
            range_all = max_all - min_all
            pos = range_all == 0
            range_all[pos] = 1
            arr -= min_all
            arr /= range_all

        return arr

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
    def cluster_score(arr, labels, outliers=False):
        """
        Modified silhouette score that excludes outliers

        Parameters
        ----------
        array : ndarray
            Array to be used for clustering, shape n samples with m features
        labels : ndarray
            Vector of cluster labels for each row in array
        outliers : bool
            Boolean flag to use outliers in silhouette_score calculation

        Returns
        -------
        s : float
            Silhouette score computed after removing outliers (label < 0)
        """
        n_clusters = len([_l for _l in np.unique(labels) if _l >= 0])
        if not outliers and n_clusters > 1:
            pos = labels >= 0
            arr = arr[pos]
            labels = labels[pos]

        s = silhouette_score(arr, labels)

        return s

    def get_data(self, cols, normalize=True, noise=0.01, **kwargs):
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

        if noise:
            arr[:, 0] *= np.random.uniform(1 - noise, 1 + noise, len(arr))

        return arr

    @staticmethod
    def optimal_eps(array, k=5):
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
        k_dist = Cluster.n_dist(array, k)
        eps_dist = Cluster.line_dist(k_dist)
        eps_pos = np.argmax(eps_dist)
        eps = k_dist[eps_pos]
        return eps

    def _cluster(self, array, min_samples, eps=None):
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

    def optimize_clusters(self, min_samples, dt=0.1, **kwargs):
        """
        Incrimentally increase eps from given value to optimize cluster
        size

        Parameters
        ----------
        min_samples : int
            min_samples value for clustering, if None set as len(array) / 1000
        dt : float
            Percentage eps by which it is to be incrementally increased
        kwargs : dict
            Internal kwargs

        Returns
        ---------
        labels : ndarray
            Vector of cluster labels for each row in array
        eps : float
            eps value used for clustering
        min_samples : int
            min_samples value used for clustering
        """
        array = self.get_data(['load', 'heat_rate'])

        labels, eps, _ = self._cluster(array, min_samples)
        score = self.cluster_score(array, labels, **kwargs)
        cluster_params = labels, eps, min_samples
        while True:
            eps_dt = eps * dt
            eps = eps + eps_dt
            labels, _, _ = self._cluster(array, min_samples, eps=eps)
            n_clusters = len(np.unique(labels))
            if n_clusters > 2:
                s = self.cluster_score(array, labels, **kwargs)
                if s >= score:
                    score = s
                    cluster_params = labels, eps, min_samples
                    logger.debug('New best fit: min_samples={}, eps={}, s={}'
                                 .format(min_samples, eps, score))
            elif n_clusters == 1:
                break

        return cluster_params

    @classmethod
    def filter(cls, unit_df, min_samples, threshold=10, **kwargs):
        """
        Parameters
        ----------
        unit_df : pandas.DataFrame
            DataFrame of timeseries heat rate data for unit of interest
        min_samples : int
            Min_samples value for clustering
        threshold : int
            Theshold for minimum number of points to filter unit
        kwargs : dict
            Internal kwargs

        Returns
        -------
        unit_df : pandas.DataFrame
            Updated DataFrame with cluster labels added for optimal cluster
        kwargs : dict
            Internal kwargs for optimize_clusters
        """
        if len(unit_df) < threshold:
            unit_df.loc[:, 'cluster'] = -1
            logger.debug('\t- Unit only has {} points and will not be filtered'
                         .format(len(unit_df)))
        else:
            cluster = cls(unit_df)
            labels, eps, min_samples = cluster.optimize_clusters(min_samples,
                                                                 **kwargs)
            logger.debug('\t- Optimal eps = {}, min_samples = {}'
                         .format(eps, min_samples))
            unit_df.loc[:, 'cluster'] = labels

        return unit_df


class SingleCluster(Cluster):
    """
    Subclass to perform single cluster extraction on non-CC generators
    """
    def __init__(self, unit_df):
        super().__init__(unit_df)
        self._tree = cKDTree(self.get_data(['load', 'heat_rate']))

    @staticmethod
    def knn(arr, tree=None, k=1):
        """
        Get euclidian distance nearest neighbors for numerical column data.

        Parameters
        ----------
        array : ndarray
            Array to be used for clustering, shape n samples with m features
        k : int
            The number of k-th nearest neighbors to return.

        Returns
        -------
        dist : np.ndarray
            Each entry gives the list of distances to the neighbors of the
            corresponding point.
        """
        # execute KNN query
        if tree is None:
            tree = cKDTree(arr)

        dist, _ = tree.query(arr, k=k)

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
        cluster = arr[labels == 0]
        noise = arr[labels == -1]
        tree = cKDTree(cluster)

        # Compute intra-cluster nearest neighbor distance
        a = tree.query(cluster, k=2)[0][:, 1:].mean()
        # Compute nearest neighbor distance between cluster and noise
        b = tree.query(noise, k=1)[0].mean()

        return b / a

    def _cluster(self, array, min_samples, eps=None, tree=False):
        """
        Find single cluster for array, compute eps if not supplied.

        Parameters
        ----------
        array : ndarray
            Array to be used for clustering, shape n samples with m features
        min_samples : int
            min_samples value for clustering
        eps : float
            Epsilon value for clustering
            If None estimate using k-n distance and min_samples
        tree : bool
            Compute tree, if False use pre-computed tree

        Returns
        ---------
        labels : ndarray
            Vector of cluster labels for each row in array
        eps : float
            eps value used to cluster
        min_samples : int
            min_samples value used to cluster
        """
        if eps is None:
            eps = self.optimal_eps(array, k=min_samples)

        n_dat = len(array)
        if min_samples > n_dat:
            raise ValueError('Number of nearest neighbors to retrieve '
                             '(min_samples={}) exceeds length of the '
                             'dataframe {}'.format(min_samples, n_dat))

        if eps is None:
            eps = self.optimal_eps(array, k=min_samples)

        logger.debug('Set contains {} coordinates, '
                     'N-neighbors required: {} at {}'
                     .format(n_dat, min_samples, eps))
        # get nearest neighbors for min_samples + 1 as there is self-reference
        if tree:
            tree = None
        else:
            tree = self._tree

        d = self.knn(array, tree=tree, k=min_samples + 1)
        # count the number of distance results per index less than the
        # threshold (exclude the 1st dist row which is self-referential)
        counts = np.sum(d[:, 1:] <= eps, axis=1)

        # find where there are NN neighbors satisfying the distance threshold
        mask = counts >= min_samples
        labels = np.full(len(array), -1)
        labels[mask] = 0

        return labels, eps, min_samples

    def optimize_clusters(self, min_samples, dt=0.1):
        """
        Incrimentally increase eps from given value to optimize cluster
        size

        Parameters
        ----------
        min_samples : int
            min_samples value for clustering, if None set as len(array) / 1000
        eps : float
            Epsilon value for clustering
            If None estimate using k-n distance and min_samples
        dt : float
            Percentage eps by which it is to be incrementally increased

        Returns
        ---------
        labels : ndarray
            Vector of cluster labels for each row in array
        eps : float
            eps value used for clustering
        min_samples : int
            min_samples value used for clustering
        """
        array = self.get_data(['load', 'heat_rate'])

        labels, eps, _ = self._cluster(array, min_samples)
        score = self.cluster_score(array, labels)
        cluster_params = labels, eps, min_samples
        while True:
            eps_dt = eps * dt
            eps = eps + eps_dt
            labels, _, _ = self._cluster(array, min_samples, eps=eps)

            if len(np.unique(labels)) == 1:
                break

            s = self.cluster_score(array, labels)
            if s < score:
                break
            else:
                score = s
                cluster_params = labels, eps, min_samples
                logger.debug('New best fit: min_samples={}, eps={}, score={}'
                             .format(min_samples, eps, score))

        return cluster_params


class ClusterCC(Cluster):
    """
    Subclass for finding operating modes in CCs
    """

    def optimize_clusters(self, min_samples, dt=0.1):
        """
        Incrimentally increase eps from given value to optimize cluster
        size

        Parameters
        ----------
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
        array = self.get_data(['load', 'heat_rate', 'cts'], norm='max')
        cts = len(np.unique(array[:, -1]))
        if cts < 2:
            # silhouette_score requires a minimum of 2 clusters
            # Assumes a minimum of 2 operating moves 1x0 and 1x1
            cts = 2

        labels, eps, _ = self._cluster(array, min_samples)
        score = self.cluster_score(array, labels)
        cluster_params = labels, eps, min_samples
        while True:
            eps_dt = eps * dt
            eps = eps + eps_dt
            labels, _, _ = self._cluster(array, min_samples, eps=eps)
            s = self.cluster_score(array, labels)
            n_clusters = len([_l for _l in np.unique(labels) if _l >= 0])
            if s < score:
                if n_clusters <= cts:
                    break
            else:
                score = s
                cluster_params = labels, eps, min_samples
                logger.debug('New best fit: min_samples={}, eps={}, score={}'
                             .format(min_samples, eps, score))

        return cluster_params
