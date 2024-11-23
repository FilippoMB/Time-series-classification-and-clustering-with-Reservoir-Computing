import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy.spatial.distance import pdist, cdist, squareform

from .reservoir import Reservoir
from .tensorPCA import tensorPCA

            
class RC_model(object):
    r"""Build and evaluate a RC-based model for time series classification or clustering.

    The training and test Multivariate Time Series (MTS) are multidimensional arrays of shape ``[N,T,V]``, where ``N`` is the number of samples, ``T`` is the number of time steps in each sample, ``V`` is the number of variables in each sample.
    
    Training and test labels have shape ``[N,C]``, with ``C`` being the number of classes.
    
    The dataset consists of training data and respective labels ``(X, Y)`` and test data and respective labels ``(Xte, Yte)``.
    
    **Reservoir parameters:**
    
    :param reservoir: object of class ``Reservoir`` (default ``None``) 
        Precomputed reservoir. If ``None``, the following structural hyperparameters must be specified.
    :param n_internal_units: int (default ``100``) 
        Processing units in the reservoir.
    :param spectral_radius: float (default ``0.99``) 
        Largest eigenvalue of the reservoir matrix of connection weights.
        To ensure the Echo State Property, set ``spectral_radius <= leak <= 1``)
    :param leak: float (default ``None``) 
        Amount of leakage in the reservoir state update.
        If ``None`` or ``1.0``, no leakage is used.
    :param connectivity: float (default ``0.3``) 
        Percentage of nonzero connection weights.
        Unused in circle reservoir.
    :param input_scaling: float (default ``0.2``)
        Scaling of the input connection weights.
        Note that the input weights are randomly drawn from ``{-1,1}``.
    :param noise_level: float (default ``0.0``) 
        Standard deviation of the Gaussian noise injected in the state update.
    :param n_drop: int (default ``0``)
        Number of transient states to drop.
    :param bidir: bool (default ``False``)
        Use a bidirectional reservoir (``True``) or a standard one (``False``).
    :param circle: bool (default ``False``)
        Generate determinisitc reservoir with circle topology where each connection 
        has the same weight.
    
    **Dimensionality reduction parameters:**
    
    :param dimred_method: str (default ``None``)
        Procedure for reducing the number of features in the sequence of reservoir states. 
        Possible options are: ``None`` (no dimensionality reduction), ``'pca'`` (standard PCA), 
        or ``'tenpca'`` (tensorial PCA for multivariate time series data).
    :param n_dim: int (default ``None``) 
        Number of resulting dimensions after the dimensionality reduction procedure.
    
    **Representation parameters:**
    
    :param mts_rep: str (default ``None``) 
        Type of MTS representation. 
        It can be ``'last'`` (last state), ``'mean'`` (mean of all the states),
        ``'output'`` (output model space), or ``'reservoir'`` (reservoir model space).
    :param w_ridge_embedding: float (default ``1.0``) 
        Regularization parameter of the ridge regression in the output model space 
        and reservoir model space representation; ignored if ``mts_rep == None``.
    
    **Readout parameters:**
    
    :param readout_type: str (default ``'lin'``) 
        Type of readout used for classification. It can be ``'lin'`` (ridge regression), 
        ``'mlp'`` (multiplayer perceptron), ``'svm'`` (support vector machine), 
        or ``None``. 
        If ``None``, the input representations will be saved in the ``.input_repr`` attribute: 
        this is useful for clustering and visualization.
        Also, if ````None````, the other Readout hyperparameters can be left unspecified.
    :param w_ridge: float (default ``1.0``) 
        Regularization parameter of the ridge regression readout (only for ``readout_type=='lin'``).
    :param mlp_layout: tuple (default ``None``) 
        Tuple with the sizes of MLP layers, e.g., ``(20, 10)`` defines a MLP with 2 layers of 20 and 10 units respectively.
        Used only when ``readout_type=='mlp'``.
    :param num_epochs: int (default ``None``) 
        Number of iterations during the optimization.
        Used only when ``readout_type=='mlp'``.
    :param w_l2: float (default ``None``) 
        Weight of the L2 regularization.
        Used only when ``readout_type=='mlp'``.
    :param nonlinearity: str (default ``None``) 
        Type of activation function ``{'relu', 'tanh', 'logistic', 'identity'}``.
        Used only when ``readout_type=='mlp'``.
    :param svm_gamma: float (default ``1.0``) 
        Bandwidth of the RBF kernel.
        Used only when ``readout_type=='svm'``.
    :param svm_C: float (default ``1.0``) 
        Regularization for SVM hyperplane.
        Used only when ``readout_type=='svm'``.
    """
    
    def __init__(self,
              # reservoir
              reservoir=None,     
              n_internal_units=100,
              spectral_radius=0.99,
              leak=None,
              connectivity=0.3,
              input_scaling=0.2,
              noise_level=0.0,
              n_drop=0,
              bidir=False,
              circle=False,
              # dim red
              dimred_method=None, 
              n_dim=None,
              # representation
              mts_rep='mean',
              w_ridge_embedding=1.0,
              # readout
              readout_type='lin',               
              w_ridge=1.0,              
              mlp_layout=None,
              num_epochs=None,
              w_l2=None,
              nonlinearity=None, 
              svm_gamma=1.0,
              svm_C=1.0):

        self.n_drop=n_drop
        self.bidir=bidir
        self.dimred_method=dimred_method
        self.mts_rep=mts_rep
        self.readout_type=readout_type
        self.svm_gamma=svm_gamma
                        
        # Initialize reservoir
        if reservoir is None:
            self._reservoir = Reservoir(n_internal_units=n_internal_units,
                                  spectral_radius=spectral_radius,
                                  leak=leak,
                                  connectivity=connectivity,
                                  input_scaling=input_scaling,
                                  noise_level=noise_level,
                                  circle=circle)
        else:
            self._reservoir = reservoir
                
        # Initialize dimensionality reduction method
        if dimred_method is not None:
            if dimred_method.lower() == 'pca':
                self._dim_red = PCA(n_components=n_dim)            
            elif dimred_method.lower() == 'tenpca':
                self._dim_red = tensorPCA(n_components=n_dim)
            else:
                raise RuntimeError('Invalid dimred method ID')
                
        # Initialize ridge regression model
        if mts_rep=='output' or mts_rep=='reservoir':
            self._ridge_embedding = Ridge(alpha=w_ridge_embedding, fit_intercept=True)
                        
        # Initialize readout type            
        if self.readout_type is not None:
            
            if self.readout_type == 'lin': # Ridge regression
                self.readout = Ridge(alpha=w_ridge)        
            elif self.readout_type == 'svm': # SVM readout
                self.readout = SVC(C=svm_C, kernel='precomputed')          
            elif readout_type == 'mlp': # MLP (deep readout)  
                # pass
                self.readout = MLPClassifier(
                    hidden_layer_sizes=mlp_layout, 
                    activation=nonlinearity, 
                    alpha=w_l2,
                    batch_size=32, 
                    learning_rate='adaptive', # 'constant' or 'adaptive'
                    learning_rate_init=0.001, 
                    max_iter=num_epochs, 
                    early_stopping=False, # if True, set validation_fraction > 0
                    validation_fraction=0.0 # used for early stopping
                    )
            else:
                raise RuntimeError('Invalid readout type')  
        
        
    def fit(self, X, Y=None, verbose=True):
        r"""Train the RC model.

        Parameters:
        -----------
        X : np.ndarray 
            Array of of shape ``[N, T, V]`` representin the training data.

        Y : np.ndarray 
            Array of shape ``[N, C]`` representing the target values.

        verbose : bool
            If ``True``, print the training time.

        Returns:
        --------
        None
        """
                
        time_start = time.time()
        
        # ============ Compute reservoir states ============ 
        res_states = self._reservoir.get_states(X, n_drop=self.n_drop, bidir=self.bidir)
        
        # ============ Dimensionality reduction of the reservoir states ============  
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                # matricize
                N_samples = res_states.shape[0]
                res_states = res_states.reshape(-1, res_states.shape[2])                   
                # ..transform..
                red_states = self._dim_red.fit_transform(res_states)          
                # ..and put back in tensor form
                red_states = red_states.reshape(N_samples,-1,red_states.shape[1])          
            elif self.dimred_method.lower() == 'tenpca':
                red_states = self._dim_red.fit_transform(res_states)       
        else: # Skip dimensionality reduction
            red_states = res_states

        # ============ Generate representation of the MTS ============
        coeff_tr = []
        biases_tr = []   
        
        # Output model space representation
        if self.mts_rep=='output':
            if self.bidir:
                X = np.concatenate((X,X[:, ::-1, :]),axis=2)                
                
            for i in range(X.shape[0]):
                self._ridge_embedding.fit(red_states[i, 0:-1, :], X[i, self.n_drop+1:, :])
                coeff_tr.append(self._ridge_embedding.coef_.ravel())
                biases_tr.append(self._ridge_embedding.intercept_.ravel())
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
            
        # Reservoir model space representation
        elif self.mts_rep=='reservoir':
            for i in range(X.shape[0]):
                self._ridge_embedding.fit(red_states[i, 0:-1, :], red_states[i, 1:, :])
                coeff_tr.append(self._ridge_embedding.coef_.ravel())
                biases_tr.append(self._ridge_embedding.intercept_.ravel())
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
        
        # Last state representation        
        elif self.mts_rep=='last':
            input_repr = red_states[:, -1, :]
            
        # Mean state representation        
        elif self.mts_rep=='mean':
            input_repr = np.mean(red_states, axis=1)
            
        else:
            raise RuntimeError('Invalid representation ID')            
            
        # ============ Train readout ============
        if self.readout_type == None: # Just store the input representations
            self.input_repr = input_repr
            
        elif self.readout_type == 'lin': # Ridge regression
            self.readout.fit(input_repr, Y)          
            
        elif self.readout_type == 'svm': # SVM readout
            Ktr = squareform(pdist(input_repr, metric='sqeuclidean')) 
            Ktr = np.exp(-self.svm_gamma*Ktr)
            self.readout.fit(Ktr, np.argmax(Y,axis=1))
            self.input_repr_tr = input_repr # store them to build test kernel
            
        elif self.readout_type == 'mlp': # MLP (deep readout)
            self.readout.fit(input_repr, Y)
                        
        if verbose:
            tot_time = (time.time()-time_start)/60
            print(f"Training completed in {tot_time:.2f} min")

            
    def predict(self, Xte):
        r"""Computes predictions for out-of-sample (test) data.

        Parameters:
        -----------
        Xte : np.ndarray
            Array of shape ``[N, T, V]`` representing the test data.

        Returns:
        --------
        pred_class : np.ndarray
            Array of shape ``[N]`` representing the predicted classes.
        """

        # ============ Compute reservoir states ============
        res_states_te = self._reservoir.get_states(Xte, n_drop=self.n_drop, bidir=self.bidir) 
        
        # ============ Dimensionality reduction of the reservoir states ============ 
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                # matricize
                N_samples_te = res_states_te.shape[0]
                res_states_te = res_states_te.reshape(-1, res_states_te.shape[2])                    
                # ..transform..
                red_states_te = self._dim_red.transform(res_states_te)            
                # ..and put back in tensor form
                red_states_te = red_states_te.reshape(N_samples_te,-1,red_states_te.shape[1])            
            elif self.dimred_method.lower() == 'tenpca':
                red_states_te = self._dim_red.transform(res_states_te)        
        else: # Skip dimensionality reduction
            red_states_te = res_states_te             
        
        # ============ Generate representation of the MTS ============
        coeff_te = []
        biases_te = []   
        
        # Output model space representation
        if self.mts_rep=='output':
            if self.bidir:
                Xte = np.concatenate((Xte,Xte[:, ::-1, :]),axis=2)  
                    
            for i in range(Xte.shape[0]):
                self._ridge_embedding.fit(red_states_te[i, 0:-1, :], Xte[i, self.n_drop+1:, :])
                coeff_te.append(self._ridge_embedding.coef_.ravel())
                biases_te.append(self._ridge_embedding.intercept_.ravel())
            input_repr_te = np.concatenate((np.vstack(coeff_te), np.vstack(biases_te)), axis=1)
        
        # Reservoir model space representation
        elif self.mts_rep=='reservoir':    
            for i in range(Xte.shape[0]):
                self._ridge_embedding.fit(red_states_te[i, 0:-1, :], red_states_te[i, 1:, :])
                coeff_te.append(self._ridge_embedding.coef_.ravel())
                biases_te.append(self._ridge_embedding.intercept_.ravel())
            input_repr_te = np.concatenate((np.vstack(coeff_te), np.vstack(biases_te)), axis=1)
    
        # Last state representation        
        elif self.mts_rep=='last':
            input_repr_te = red_states_te[:, -1, :]
            
        # Mean state representation        
        elif self.mts_rep=='mean':
            input_repr_te = np.mean(red_states_te, axis=1)
            
        else:
            raise RuntimeError('Invalid representation ID')   
            
        # ============ Apply readout ============
        if self.readout_type == 'lin': # Ridge regression        
            logits = self.readout.predict(input_repr_te)
            pred_class = np.argmax(logits, axis=1)
            
        elif self.readout_type == 'svm': # SVM readout
            Kte = cdist(input_repr_te, self.input_repr_tr, metric='sqeuclidean')
            Kte = np.exp(-self.svm_gamma*Kte)
            pred_class = self.readout.predict(Kte)
            
        elif self.readout_type == 'mlp': # MLP (deep readout)
            pred_class = self.readout.predict(input_repr_te)
            pred_class = np.argmax(pred_class, axis=1)
            
        return pred_class
    

class RC_forecaster(object):
    r"""Class to perform time series forecasting with RC.

    The training and test data are multidimensional arrays of shape ``[T,V]``, with

    - ``T`` = number of time steps in each sample,
    - ``V`` = number of variables in each sample.

    Given a time series ``X``, the training data are supposed to be as follows:
    
        ``Xtr, Ytr = X[0:-forecast_horizon,:], X[forecast_horizon:,:]``

    Once trained, the model can be used to compute prediction ``forecast_horizon`` steps ahead:
        
            ``Yhat[t,:] = Xte[t+forecast_horizon,:]``

    **Reservoir parameters:**

    :param reservoir: object of class ``Reservoir`` (default ``None``)
        Precomputed reservoir. If ``None``, the following structural hyperparameters must be specified.
    :param n_internal_units: int (default ``100``) 
        Processing units in the reservoir.
    :param spectral_radius: float (default ``0.99``) 
        Largest eigenvalue of the reservoir matrix of connection weights.
        To ensure the Echo State Property, set ``spectral_radius <= leak <= 1``)
    :param leak: float (default ``None``) 
        Amount of leakage in the reservoir state update.
    :param connectivity: float (default ``0.3``)
        Percentage of nonzero connection weights.
    :param input_scaling: float (default ``0.2``) 
        Scaling of the input connection weights.
        Note that the input weights are randomly drawn from ``{-1,1}``.
    :param noise_level: float (default ``0.0``)
        Standard deviation of the Gaussian noise injected in the state update.
    :param n_drop: int (default ``0``)
        Number of transient states to drop.
    :param circle: bool (default ``False``)
        Generate determinisitc reservoir with circle topology where each connection 
        has the same weight.

    **Dimensionality reduction parameters:**

    :param dimred_method: str (default ``None``)
        Procedure for reducing the number of features in the sequence of reservoir states. 
        Possible options are: ``None`` (no dimensionality reduction), ``'pca'`` (standard PCA), 
        or ``'tenpca'`` (tensorial PCA for multivariate time series data).
    :param n_dim: int (default ``None``) 
        Number of resulting dimensions after the dimensionality reduction procedure.

    **Readout parameters:**

    :param w_ridge: float (default ``1.0``) 
        Regularization parameter of the ridge regression readout.
    """
    
    def __init__(self,
                # reservoir
                reservoir=None,     
                n_internal_units=100,
                spectral_radius=0.99,
                leak=None,
                connectivity=0.3,
                input_scaling=0.2,
                noise_level=0.0,
                n_drop=0,
                circle=False,
                # dim red
                dimred_method=None, 
                n_dim=None,
                # readout              
                w_ridge=1.0):
        self.n_drop=n_drop
        self.dimred_method=dimred_method  
                        
        # Initialize reservoir
        if reservoir is None:
            self._reservoir = Reservoir(n_internal_units=n_internal_units,
                                        spectral_radius=spectral_radius,
                                        leak=leak,
                                        connectivity=connectivity,
                                        input_scaling=input_scaling,
                                        noise_level=noise_level,
                                        circle=circle)
        else:
            self._reservoir = reservoir
                
        # Initialize dimensionality reduction method
        if dimred_method is not None:
            if dimred_method.lower() == 'pca':
                self._dim_red = PCA(n_components=n_dim)            
            else:
                raise RuntimeError('Invalid dimred method ID')
            
        # Initialize readout
        self.readout = Ridge(alpha=w_ridge)


    def fit(self, X, Y, verbose=True):
        r"""Train the RC model for forecasting.

        Parameters:
        -----------
        X : np.ndarray 
            Array of shape ``[T, V]`` representing the training data.

        Y : np.ndarray
            Array of shape ``[T, V]`` representing the target values.

        verbose : bool
            If ``True``, print the training time.

        Returns:
        --------
        red_states : np.ndarray
            Array of shape ``[T, n_dim]`` representing the reservoir states of the time steps used for training.
        """
        
        time_start = time.time()
        
        # ============ Compute reservoir states ============ 
        res_states = self._reservoir.get_states(X[None,:,:], n_drop=self.n_drop, bidir=False)
        
        # ============ Dimensionality reduction of the reservoir states ============  
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                red_states = self._dim_red.fit_transform(res_states[0])          
        else: # Skip dimensionality reduction
            red_states = res_states[0]

        self._fitted_states = red_states

        # ============ Train readout ============
        self.readout.fit(red_states, Y[self.n_drop:,:])          
            
        if verbose:
            tot_time = (time.time()-time_start)/60
            print(f"Training completed in {tot_time:.2f} min")

        return red_states

    def predict(self, Xte, return_states=False):
        r"""Computes predictions for out-of-sample (test) data.

        Parameters:
        -----------
        Xte : np.ndarray
            Array of shape ``[T, V]`` representing the test data.
        
        return_states : bool
            If ``True``, return the predicted states.

        Returns:
        --------
        Yhat : np.ndarray
            Array of shape ``[T, V]`` representing the predicted values.
        
        red_states_te : np.ndarray
            Array of shape ``[T, n_dim]`` representing the reservoir states of the new time steps.
        """

        # ============ Compute reservoir states ============
        res_states_te = self._reservoir.get_states(Xte[None,:,:], n_drop=self.n_drop, bidir=False) 
        
        # ============ Dimensionality reduction of the reservoir states ============ 
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                red_states_te = self._dim_red.transform(res_states_te[0])                          
        else: # Skip dimensionality reduction
            red_states_te = res_states_te[0]        

        self._predicted_states = red_states_te

        # ============ Apply readout ============
        Yhat = self.readout.predict(red_states_te)

        if return_states:
            return Yhat, red_states_te
        return Yhat

    def get_fitted_states(self):
        r"""Return the fitted reservoir states.

        Returns:
        --------
        fitted_states : np.ndarray
            Array of shape ``[T, n_dim]`` representing the fitted reservoir states.
        """
        return self._fitted_states

    def get_predicted_states(self):
        r"""Return the predicted reservoir states.

        Returns:
        --------
        predicted_states : np.ndarray
            Array of shape ``[T, n_dim]`` representing the predicted reservoir states.
        """
        return self._predicted_states
