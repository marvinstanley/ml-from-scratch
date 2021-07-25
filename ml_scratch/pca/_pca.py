import numpy as np
import warnings

class PCA():
    """
    Principal component analysis (PCA).

    It uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.
    This implementation uses eigen decomposition on the covariance matrix rather than Singular Value Decomposition (SVD)
    
    By default, values fitted are centered on 0 and scaled (standarized)


    Parameters
    ----------
        n_components : int
            Number of Principal Components that want to be kept:: 
            
            min(n_components, num_features)

        decomposition : {‘eigen’, ‘svd’}, default=’eigen’
            The method of decomposition.
            'eigen' uses the Eigenvalue and Eigenvector Decomposition
            'svd' uses the Singular Value Decomposition (SVD)

    Attributes
    ----------
        components_ : numpy.array of shape(n_components, num_features)
            The Principal Components that are kept, based on ``n_components``.
            Components are sorted based on ``explained_variance_``

        explained_variance_ : numpy.array of shape(n_components)
            The explained variance of the associated components

        explained_variance_ratio_ : numpy.array of shape(n_components)
            The ratio of the explained variance
    """
    def __init__(self, n_components: int = None, method: str = 'eigen'):
        self.n_components = int(n_components)
        self.method = method

    def _get_covariance(self, x):
        """
        Calculate the covariance of the matrix. Not meant to be used explicitly

        .. note::
            numpy.cov() considers its input data matrix to have observations in each column,
            and variables in each row.

            Either set rowvar=False or transpose X

        Parameters
        ----------
            x : numpy.array of shape(num_samples, num_features)
                Matrix to be fed to find the covariance

        Returns
        ----------
            x : numpy.array of shape(num_samples, num_features)
                Matrix which values has been standarized
        """
        # Why you have to transpose the matrix for numpy.cov()
        # https://stats.stackexchange.com/a/263508
        x = np.cov(x.T, bias=True)
        return x

    def _standarize(self, x : np.array):
        """
        Standarize the matrix. Not meant to be used explicitly.

        Parameters
        ----------
            x : numpy.array of shape(num_samples, num_features)
                Matrix to be standarized

        Returns
        ----------
            x : numpy.array of shape(num_samples, num_features)
                Matrix which values has been standarized
        """
        
        self.mean_ = np.mean(x, axis=0)

        # Using (N-1) for sample size
        self.std_ = np.std(x, axis=0, ddof=1)

        # Standarize matrix (x - mean) / standard_deviation
        x = (x - self.mean_) # / self.std_

        return x

    def _get_eigen(self, x):
        """
        Finding Eigenvalues and Eigenvectors in order to get ``explained_variance`` and ``components_``.
        Not to be used explicitly

        Parameters
        ----------
            x : numpy.array of shape(num_features, num_features)
                Matrix to be standarized

        Returns
        ----------
            val : numpy.array of shape(num_features)
                Explained Variance of the Principal Components. This is the sorted EigenValue
            vec : numpy.array of shape(num_features, num_features)
                The Principal Components of the vectors. This is the sorted EigenVectors based on the EigenValue
        """
        # Since matrix from covariance is always symmetrical
        # We can immediately use complex Hermitian (conjugate symmetric)
        # Eigh already sorted the eigen value and vectors
        val, vec = np.linalg.eigh(x)

        # reverse the order of the eigen value
        val = val[::-1]
        vec = vec.T[::-1]

        return val, vec

    def _get_svd(self, x):
        """
        Using SVD to get the ``components_`` and ``explained_variance_``.
        Only the matrix Sigma and Vh is going to be used

        Parameters
        ----------
            x : numpy.array of shape(num_features, num_features)
                Matrix to be standarized

        Returns
        ----------
            Sigma : numpy.array of shape(num_features)
                Explained Variance of the Principal Components. This is matrix Sigma or Σ obtained from SVD
            Vh : numpy.array of shape(num_features, num_features)
                The Principal Components of the vectors. This is matrix Vh or V* obtained from SVD
        """

        U, Sigma, Vh = np.linalg.svd(x, full_matrices=False)

        Sigma = (Sigma ** 2) / (4)

        # sort the value based on explained_variance
        Sigma_ind = np.argsort(Sigma)

        Sigma = Sigma[Sigma_ind[::-1]]
        Vh = Vh[Sigma_ind[::-1]]

        return Sigma, Vh

    def fit(self, x: np.array, limit_var_ratio: bool = False, ratio_threshold: int = 0.6):
        """
        Fit matrix X to the model to get the Principal Components.
        
        By default, values fitted are centered on 0 and scaled (standarized)

        Parameters
        ----------
            x : numpy.array of shape(num_samples, num_features)
                Matrix to be fit to the model
            limit_var_ratio: bool, optional, default=False
                Whether or not to set the components to be kept based on the variance ratio
            ratio_threshold : float, optional, default=0.6
        """

        self._feature_dim = x.shape[1]

        assert self._feature_dim > 1, "Dimension of array must be greater than (n, 1)"

        assert len(x.shape) == 2, "Expecting 2D array, got {}D array instead".format(len(x.shape))

        if(x.shape[1] < self.n_components):
            warnings.warn("Feature dimension of the array is smaller than n_components, changing n_components to feature dimension...")
        
        n_components = min(self.n_components, x.shape[1])

        # Standarize the matrix first
        x = self._standarize(x)

        if self.method == 'eigen':
            # Get the covariance of the matrix
            x = self._get_covariance(x)

            # Get the eigenvalue and eigenvectors (components_ and explained_variance_ in sklearn)
            self.explained_variance_, self.components_ = self._get_eigen(x)
        
        elif self.method == 'svd':
            print('using SVD')
            self.explained_variance_, self.components_ = self._get_svd(x)

        # Calculate variance ratio
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)

        # Set components that are kept
        self.explained_variance_ = self.explained_variance_[:n_components]
        self.components_ = self.components_[:n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_components]


    def transform(self, x):
        """
        Apply feature transformation or dimensionality reduction to matrix X

        Parameters
        ----------
            x : numpy.array of shape(num_samples, num_features)
                Matrix to be transformed
        
        Returns
        ----------
            numpy.array of shape(num_samples, n_components)
            Matrix that has undergone dimensionality reduction
        """
        assert self.components_.any(), 'Principal components have not been calculated, run PCA.fit() first'

        assert x.shape[1] == self._feature_dim, "Feature Dimension of X is mismatched from the fitted model"

        # Standarize the matrix
        x = self._standarize(x)

        # Transpose the components
        return np.matmul(x, self.components_.T)

    def fit_transform(self, x):
        """
        Fit matrix X to the model to get the Principal Components and
        apply feature transformation or dimensionality reduction to matrix X

        Parameters
        ----------
            x : numpy.array of shape(num_samples, num_features)
                Matrix to be transformed
        
        Returns
        ----------
            numpy.array of shape(num_samples, n_components)
            Matrix that has undergone dimensionality reduction
        """
        self.fit(x)
        return self.transform(x)

