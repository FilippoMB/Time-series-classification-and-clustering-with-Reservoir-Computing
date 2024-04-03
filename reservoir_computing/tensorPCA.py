import numpy as np
import numpy.linalg as linalg


class tensorPCA:
    r"""
    Compute PCA on a dataset of multivariate time series represented as a 3-dimensional tensor
    and reduce the size along the third dimension from ``[N, T, V]`` to ``[N, T, D]``, where ``D <= V`` .

    The input dataset must be a 3-dimensional tensor, where the first dimension ``N`` represents 
    the number of observations, the second dimension ``T`` represents the number of time steps 
    in the time series, and the third dimension ``V`` represents the number of variables in the time series.

    Parameters
    ----------
    n_components : int
        The number of principal components to keep after the dimensionality reduction. This
        determines the size of the third dimension ``D`` in the output tensor.
    """

    def __init__(self, n_components):
        self.n_components=n_components
        self.first_eigs = None
        
    def fit(self, X):
        r"""
        Fit the tensorPCA model to the input dataset ``X``.
        
        Parameters:
        ------------
        X : np.ndarray
            Time series, 3D array of shape ``[N,T,V]``, where ``N`` is the number of time series,
            ``T`` is the length of each time series, and ``V`` is the number of variables in each.

        Returns:
        ------------
        None
        """
        if len(X.shape) != 3:
            raise RuntimeError('Input must be a 3d tensor')
        
        Xt = np.swapaxes(X,1,2)  # [N,T,V] --> [N,V,T]
        Xm = np.expand_dims(np.mean(X, axis=0), axis=0) # mean sample
        Xmt = np.swapaxes(Xm,1,2)
        
        C = np.tensordot(X-Xm,Xt-Xmt,axes=([1,0],[2,0])) / (X.shape[0]-1) # covariance of 0-mode slices
        
        # Sort eigenvalues of covariance matrix
        eigenValues, eigenVectors = linalg.eig(C)
        idx = eigenValues.argsort()[::-1]   
        eigenVectors = eigenVectors[:,idx]
        
        self.first_eigs = eigenVectors[:,:self.n_components]
        
    def transform(self, X):
        r"""
        Transform the input dataset X using the tensorPCA model.

        Parameters:
        ------------
        X : np.ndarray
            Time series, 3D array of shape ``[N,T,V]``, where ``N`` is the number of time series,
            ``T`` is the length of each time series, and ``V`` is the number of variables in each.

        Returns:
        ------------
        Xpca : np.ndarray
            Transformed time series, 3D array of shape ``[N,T,D]``, where ``N`` is the number of time series,
            ``T`` is the length of each time series, and ``D`` is the number of principal components.
        """
        return np.einsum('klj,ji->kli',X,self.first_eigs)
    
    def fit_transform(self, X):
        r"""
        Fit the tensorPCA model to the input dataset ``X`` and transform it.

        Parameters:
        ------------
        X : np.ndarray
            Time series, 3D array of shape ``[N,T,V]``, where ``N`` is the number of time series,
            ``T`` is the length of each time series, and ``V`` is the number of variables in each.

        Returns:
        ------------
        Xpca : np.ndarray
            Transformed time series, 3D array of shape ``[N,T,D]``, where ``N`` is the number of time series,
            ``T`` is the length of each time series, and ``D`` is the number of principal components.
        """
        self.fit(X)
        return self.transform(X)