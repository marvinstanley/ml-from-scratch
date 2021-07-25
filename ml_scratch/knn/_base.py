from .._distance import manhattan_distance, cosine_distance, euclidean_distance
import heapq
import numpy as np

class BaseKNN():
    """
    Base class for K Nearest Neighbor.
    Not to be used explicitly

    .. note::

        It currently only accepts numpy arrays.
        Be sure to convert your data to numpy arrays

    Parameters
    ----------
        k_neighbor : int
            K number of neighbors to be used for finding target prediction

        algorithm : {‘brute‘, ‘kdtree‘}, default='brute'
            Algorithm for finding the nearest neighbor
            
                - If ‘brute‘:
                    Use bruteforce search to find the nearest neighbors, uses priority queue with max queue of ``k_neighbor``
                - If ‘kdtree‘:
                    Use kdtree algorithm to find the nearest neighbors
            
        distance_metric: {‘manhattan‘, ‘euclidean‘, ‘cosine‘}, default='manhttan'
            Distance metric used to find the distance between samples

                - If ‘manhattan‘:
                    Uses the manhattan distance, formula:
                - If ‘euclidean‘:
                    Uses the euclidean distance, formula:
                - If ‘cosine‘:
                    Uses the cosine distance, formula:

    Attributes
    ----------
        metrics_ : numpy.array of shape(n_samples, )
            The distance metric from the test samples to the train samples
            
    """
    def __init__(self, k_neighbor : int = 5, algorithm : str = 'brute', distance_metric : str = 'manhattan'):

        assert algorithm in {'brute', 'kd-tree'}, 'Algorithm not valid, please choose between {\'brute\', \'kd-tree\'}'
        
        assert distance_metric in {'manhattan', 'euclidean', 'cosine'}, 'Distance metric not valid, please choose between {\'manhattan\', \'euclidean\', \'cosine\'}'

        self.k_neighbor = int(k_neighbor)
        self.algorithm = algorithm
        self.distance_metric = distance_metric

    def fit(self, X, y):
        """
        Fit the input sample and target sample to the model.

        Parameters
        ----------
            X : numpy.array of shape(n_samples, n_features)
                The input data of the training set
            y : numpy.array of shape(n_samples)
                The output/target data of the training set
        """

        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"

        assert y.shape[0] == 1, "y must be a 1D array, got {}D array".format(y.shape[0])

        assert X.shape[1] == 2, "X must be a 2D array, got {}D array".format(y.shape[0])

        self._fit_X = X
        self._fit_y = y

    def _calc_distance(self, X):
        """
        Calculate the distance of the input test set to the input training set.

        Parameters
        ----------
            X : numpy.array of shape(n_test_samples, n_features)
                The input of the testing set

        Returns
        ----------
            metric : numpy.array of shape(n_test_samples,)
                The distances of the test sample calculated from the fitted training set

        """
        assert self.distance_metric in {'manhattan', 'euclidean', 'cosine'}, 'Distance metric not valid, please choose between {\'manhattan\', \'euclidean\', \'cosine\'}'

        if self.distance_metric == 'manhattan':
            metric =  manhattan_distance(self._fit_X, X)
        elif self.distance_metric == 'cosine':
            metric =  cosine_distance(self._fit_X, X)
        elif self.distance_metric == 'euclidean':
            metric =  euclidean_distance(self._fit_X, X)

        return metric

    def _find_neighbor(self, X):
        """
        Bruteforce search of finding the nearest neighbors using priority queue

        Parameters
        ----------
            X : numpy.array of shape(n_test_samples, n_features)
                The input of the testing set

        Returns
        ----------
            neigh_dist : numpy.array of shape(k_neighbor, )
                The top ``k_neighbor`` neighbors distances from the test set to the training set
            neigh_target : numpy.array of shape(k_neighbor, )
                The top ``k_neighbor`` neighbors target value of the training set
        """
        self.metrics_ = self._calc_distance(X)

        # No need to sort the whole distance, only need to find k minimum
        # Use priority queue
        neigh_dist = []
        neigh_target = []

        for metric in self.metrics_:
            neighbors = []
            # zip metric with y
            # flip the distance because heapq is a min heap (it will pop the min value)
            for item in zip(metric * -1, self._fit_y):
                if len(neighbors) < self.k_neighbor:
                    heapq.heappush(neighbors, item)
                else:
                    heapq.heappushpop(neighbors, item)

            neigh_dist.append([n[0] for n in neighbors])
            neigh_target.append([n[1] for n in neighbors])

        return np.array(neigh_dist), np.array(neigh_target)