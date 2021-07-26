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

    .. code-block:: python

        from ml_scratch.knn import KNNRegressor
        import numpy as np

        model = KNNRegressor(3)

        A_x = np.array([[1, 2, 3, 4], 
                    [5, 5, 6, 7], 
                    [1, 4, 2, 3],
                    [3, 1, 3, 7],
                    [2, 8, 1, 9],
                    [1, 5, 9, 7]])

        # A_y = ['A', 'A', 'A', 'B', 'B', 'B']
        A_y = [1.0, 2.5, 0.6, -1.0, 5.0, 3.0]

        B_x = np.array([[5, 3, 2, 1],
                    [8, 1, 2, 2]])

        model.fit(A_x, A_y)
        print(model.predict(B_x))


    Parameters
    ----------
        k_neighbor : int
            K number of neighbors to be used for finding target prediction
            
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
        
        assert kwargs['k_neighbor'] % 2 == 1, 'k_neighbor must be an odd number'

        super(KNNClassifier, self).__init__(*args, **kwargs)

    def predict(self, X):
        """
        Predicts the output of the test set from the near neighbors.
        Counts the highest occuring target label from the near neighbors.

        Parameters
        ----------
            X : numpy.array of shape(n_test_samples, n_features)
                The input of the testing set

        Returns
        ----------
            y_pred : numpy.array of shape(n_test_samples)
                The predicted target values based on the test set
        """

        assert X.shape[1] == self._fit_X.shape[1], "Mismatched number of features from the fitted model \
        expected shape of (n_test_samples, {}) got (n_test_samples, {})".format(self._fit_X.shape[1], X.shape[1])

        assert len(X.shape) == 2, "X must be a 2D array, got {}D array".format(len(X.shape))

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

    .. code-block:: python

        # #############################################################################
        # Generate sample data
        import numpy as np
        import matplotlib.pyplot as plt
        from ml_scratch.knn import KNNRegressor

        np.random.seed(0)
        X = np.sort(5 * np.random.rand(40, 1), axis=0)
        T = np.linspace(0, 5, 500)[:, np.newaxis]
        y = np.sin(X).ravel()

        # Add noise to targets
        y[::5] += 1 * (0.5 - np.random.rand(8))

        # #############################################################################
        # Fit regression model
        n_neighbors = 10


        knn = KNNRegressor(n_neighbors)
        y_ = knn.fit(X, y).predict(T)

        plt.scatter(X, y, color='darkorange', label='data')
        plt.plot(T, y_, color='navy', label='prediction')

        plt.legend()
        plt.title("KNeighborsRegressor")

        plt.show()


    Parameters
    ----------
        k_neighbor : int
            K number of neighbors to be used for finding target prediction

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

        Parameters
        ----------
            X : numpy.array of shape(n_test_samples, n_features)
                The input of the testing set

        Returns
        ----------
            y_pred : numpy.array of shape(n_test_samples)
                The predicted target values based on the test set
        """

        assert X.shape[1] == self._fit_X.shape[1], "Mismatched number of features from the fitted model \
        expected shape of (n_test_samples, {}) got (n_test_samples, {})".format(self._fit_X.shape[1], X.shape[1])

        assert len(X.shape) == 2, "X must be a 2D array, got {}D array".format(len(X.shape))

        _, y_pred = self._find_neighbor(X)

        # Interpolate / Mean
        y_pred = np.mean(y_pred, axis=1)

        return y_pred.ravel()