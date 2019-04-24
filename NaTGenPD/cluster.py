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

        Returns
        -------
        s : float
            Silhouette score computed after removing outliers (label < 0)
        """
        if not outliers:
            pos = labels >= 0
            arr = arr[pos]
            labels = labels[pos]

        return silhouette_score(arr, labels)

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

    def optimize_clusters(self, array, min_samples, dt=0.1, **kwargs):
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
        outliers = kwargs.get('outliers', False)
        labels, eps, _ = self._cluster(array, min_samples)
        label_n = [_l for _l in np.unique(labels) if _l != -1]
        n_clusters = len(label_n)
        eps_dt = eps * dt

        score = self.cluster_score(array, labels, outliers=outliers)
        cluster_params = labels, eps, min_samples
        while len(label_n) > 1:
            eps += eps_dt
            eps_dt = eps * dt
            labels, _, _ = self._cluster(array, min_samples, eps=eps)

            label_n = [_l for _l in np.unique(labels) if _l != -1]
            if len(label_n) == n_clusters:
                s = self.cluster_score(array, labels, outliers=outliers)
                if s > score:
                    score = s
                    cluster_params = labels, eps, min_samples
            else:
                break

        return cluster_params

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
        outliers = kwargs.get('outliers', False)
        cluster_params = None
        score = None
        while True:
            labels, eps, _ = self._cluster(array, min_samples)

            label_n = [_l for _l in np.unique(labels) if _l != -1]
            if len(label_n) == n_clusters:
                if cluster_params is None:
                    cluster_params = labels, eps, min_samples
                    score = self.cluster_score(array, labels,
                                               outliers=outliers)
                else:
                    s = self.cluster_score(array, labels, outliers=outliers)
                    if s > score:
                        score = s
                        cluster_params = labels, eps, min_samples

                if optimize:
                    dt = kwargs.get('dt', 0.1)
                    cluster_params = self.optimize_clusters(array,
                                                            min_samples,
                                                            dt=dt, **kwargs)
            elif len(label_n) < n_clusters:
                if cluster_params is None:
                    raise RuntimeError('{:} clusters could not be found'
                                       .format(n_clusters))
                else:
                    break

            min_samples += 1

        return cluster_params


class SingleCluster(Cluster):
    """
    Subclass to perform single cluster extraction on non-CC generators
    """
    def __init__(self, unit_df):
        super().__init__(unit_df)
        self._tree = cKDTree(self.get_data(['load', 'heat_rate']))

    @staticmethod
    def knn(arr, tree=None, k=1):
        """Get euclidian distance nearest neighbors for numerical column data.

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

        dist, ind = tree.query(arr, k=k)

        logger.debug('KNN index nearest neighbors: \n{}'.format(ind))
        logger.debug('KNN distance: \n{}'.format(dist))

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
        cluster = arr[labels == 1]
        noise = arr[labels == 0]
        tree = cKDTree(cluster)

        # Compute intra-cluster nearest neighbor distance
        a = tree.query(cluster, k=2)[0][:, 1:].mean()
        # Compute nearest neighbor distance between cluster and noise
        b = tree.query(noise, k=1)[0].mean()

        return b / a

    def _cluster(self, array, min_samples, eps=None, tree=False):
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
        tree : bool
            Compute tree, if False use pre-computed tree

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
        labels = np.zeros(len(array))
        labels[mask] = 1

        return labels, eps, min_samples

    def optimize_clusters(self, array, min_samples=None, dt=0.1, **kwargs):
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
        if min_samples is None:
            min_samples = int(len(array) / 1000)

        labels, eps, _ = self._cluster(array, min_samples)
        score = self.cluster_score(array, labels)
        cluster_params = labels, eps, min_samples
        while True:
            eps_dt = eps * dt
            eps = eps + eps_dt
            labels, _, _ = self._cluster(array, min_samples, eps=eps)
            s = self.cluster_score(array, labels)
            if s >= score:
                score = s
                cluster_params = labels, eps, min_samples
            else:
                break

        return cluster_params
