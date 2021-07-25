from ._base import BaseKNN
from collections import Counter
import numpy as np

class KNNClassifier(BaseKNN):
    """
    Nearest Neighbor for Categorical target values.
    Counts the highest occuring target label from the near neighbors.

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
            Distance metric used to find the distance between samples. See ml_scratch._distance for more info

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
    def __init__(self, *args, **kwargs):
        super(KNNClassifier, self).__init__(*args, **kwargs)

    def predict(self, X):
        """
        Predicts the output of the test set from the near neighbors.
        Counts the highest occuring target label from the near neighbors.

        Parameters
        ----------
            X : numpy.array of shape(n_test_samples, n_features)
                The input of the testing set

        
        """

        assert X.shape[1] == self._fit_X.shape[1], "Mismatched number of features from the fitted model \
        expected shape of (n_test_samples, {}) got (n_test_samples, {})".format(self._fit_X.shape[1], X.shape[1])

        _, y_pred = self._find_neighbor(X)

        # Find max occurance of label
        y_pred = [Counter(p).most_common(1)[0][0] for p in y_pred]

        return np.array(y_pred).ravel()

class KNNRegressor(BaseKNN):
    """
    Nearest Neighbor for continous target values.
    Uses interpolation/mean to find the nearest value from the near neighbors.

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
            Distance metric used to find the distance between samples. See ml_scratch._distance for more info

                - If ‘manhattan‘:
                    Uses the manhattan distance
                - If ‘euclidean‘:
                    Uses the euclidean distance
                - If ‘cosine‘:
                    Uses the cosine distance

    Attributes
    ----------
        metrics_ : numpy.array of shape(n_samples, )
            The distance metric from the test samples to the train samples
    """
    def __init__(self, *args, **kwargs):
        super(KNNRegressor, self).__init__(*args, **kwargs)

    def predict(self, X):
        """
        Predicts the output of the test set from the near neighbors.
        Uses interpolation/mean to find the nearest value from the near neighbors.
        """

        assert X.shape[1] == self._fit_X.shape[1], "Mismatched number of features from the fitted model \
        expected shape of (n_test_samples, {}) got (n_test_samples, {})".format(self._fit_X.shape[1], X.shape[1])

        _, y_pred = self._find_neighbor(X)

        # Interpolate / Mean
        y_pred = np.mean(y_pred, axis=1)

        return y_pred.ravel()