import numpy as np
import numpy.linalg as linalg

class tensorPCA:

    def __init__(self, n_components):
        """
        Compute PCA on a dataset of multivariate time series represented as 3-dimensional tensor
        and reduces the size along the third dimension:
            [N, T, V] --> [N, T, D]
        with D <= V.
        The input dataset must be a 3-dimensional tensor with shapes
            - N: number of observations
            - T: number of time steps in the time series
            - V: number of variables in the time series

        Parameters:
        ----------
        n_components = int
            Number of components to keep after the dimensionality reduction.
        """
        self.n_components=n_components
        self.first_eigs = None
        
    def fit(self, X):
        """
        Fit the tensorPCA model to the input dataset X.
        
        Parameters:
        ----------
        X = array
            Time series, 3D array of shape (N,T,V), where N is the number of time series,
            T is the length of each time series, and V is the number of variables in each.
        """
        if len(X.shape) is not 3:
            raise RuntimeError('Input must be a 3d tensor')
        
        Xt = np.swapaxes(X,1,2)  # [N,T,V] --> [N,V,T]
        Xm = np.expand_dims(np.mean(X, axis=0), axis=0) # mean sample
        Xmt = np.swapaxes(Xm,1,2)
        
        C = np.tensordot(X-Xm,Xt-Xmt,axes=([1,0],[2,0])) / (X.shape[0]-1) # covariance of 0-mode slices
        
        # sort eigenvalues of covariance matrix
        eigenValues, eigenVectors = linalg.eig(C)
        idx = eigenValues.argsort()[::-1]   
        eigenVectors = eigenVectors[:,idx]
        
        self.first_eigs = eigenVectors[:,:self.n_components]
        
    def transform(self, X):
        return np.einsum('klj,ji->kli',X,self.first_eigs)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
