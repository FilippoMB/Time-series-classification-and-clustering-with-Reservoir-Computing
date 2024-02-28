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

    """
    Build and evaluate a RC-based model for time series classification or clustering.
    The training and test MTS are multidimensional arrays of shape [N,T,V], with
        - N = number of samples
        - T = number of time steps in each sample
        - V = number of variables in each sample
    Training and test labels have shape [N,C], with C the number of classes
    
    The dataset consists of:
        X, Y = training data and respective labels
        Xte, Yte = test data and respective labels
        
    Reservoir parameters:
        reservoir = precomputed reservoir (oject of class 'Reservoir');
            if None, the following structural hyperparameters must be specified
        n_internal_units = processing units in the reservoir
        spectral_radius = largest eigenvalue of the reservoir matrix of connection weights
        leak = amount of leakage in the reservoir state update (optional)
        connectivity = percentage of nonzero connection weights
        input_scaling = scaling of the input connection weights
        noise_level = deviation of the Gaussian noise injected in the state update
        n_drop = number of transient states to drop
        bidir = use a bidirectional reservoir (True or false)
            
    Dimensionality reduction parameters:
        dimred_method = procedure for reducing the number of features in the sequence of reservoir states;
            possible options are: None (no dimensionality reduction), 'pca' or 'tenpca'
        n_dim = number of resulting dimensions after the dimensionality reduction procedure
        
    Representation parameters:
        mts_rep = type of MTS representation. It can be 'last' (last state), 'output' (output model space),
            or 'reservoir' (reservoir model space)
        w_ridge_embedding = regularization parameter of the ridge regression in the output model space
            and reservoir model space representation; ignored if mts_rep == None
        
    Readout parameters:
        readout_type = type of readout used for classification. It can be 'lin' (ridge regression), 
            'mlp' (multiplayer perceptron), 'svm' (support vector machine), or None.
            If None, the input representations will be saved instead: this is useful for clustering and visualization.
        w_ridge = regularization parameter of the ridge regression readout (only for readout_type=='lin')              
        mlp_layout = tuple with the sizes of MLP layers, e.g. (20, 10) defines a MLP with 2 layers 
            of 20 and 10 units respectively. (only for readout_type=='mlp')
        num_epochs = number of iterations during the optimization (only for readout_type=='mlp')
        w_l2 = weight of the L2 regularization (only for readout_type=='mlp')
        nonlinearity = type of activation function {'relu', 'tanh', 'logistic', 'identity'} (only for readout_type=='mlp')
        svm_gamma = bandwith of the RBF kernel (only for readout_type=='svm')
        svm_C = regularization for SVM hyperplane (only for readout_type=='svm')
    """
    
    def __init__(self,
              # reservoir
              reservoir=None,     
              n_internal_units=None,
              spectral_radius=None,
              leak=None,
              connectivity=None,
              input_scaling=None,
              noise_level=None,
              n_drop=None,
              bidir=False,
              circle=False,
              # dim red
              dimred_method=None, 
              n_dim=None,
              # representation
              mts_rep=None,
              w_ridge_embedding=None,
              # readout
              readout_type=None,               
              w_ridge=None,              
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
        """
        Train the RC model.

        Parameters
        ----------
        X : array-like, shape [N, T, V]
            Training data.

        Y : array-like, shape [N, C]
            Target values.
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
        """
        Computes predictions for out-of-sample (test) data.

        Parameters
        ----------
        Xte : array-like, shape [N, T, V]
            Test data.


        Returns
        -------
        pred_class : array, shape [N]
            Predicted classes.
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
    """
    Class to perform time series forecasting with RC.

    The training and test data are multidimensional arrays of shape [T,V], with
        - T = number of time steps in each sample
        - V = number of variables in each sample
    
    Given a time series X, the training data are supposed to be as follows:
        Xtr, Ytr = X[0:-forecast_horizon,:], X[forecast_horizon:,:]

    Once trained, the model can be used to compute prediction forecast_horizon steps ahead:
        Yhat[t,:] = Xte[t+forecast_horizon,:]
        
    Reservoir parameters:
        reservoir = precomputed reservoir (oject of class 'Reservoir');
            if None, the following structural hyperparameters must be specified
        n_internal_units = processing units in the reservoir
        spectral_radius = largest eigenvalue of the reservoir matrix of connection weights
        leak = amount of leakage in the reservoir state update (optional)
        connectivity = percentage of nonzero connection weights
        input_scaling = scaling of the input connection weights
        noise_level = deviation of the Gaussian noise injected in the state update
        n_drop = number of transient states to drop
            
    Dimensionality reduction parameters:
        dimred_method = procedure for reducing the number of features in the sequence of reservoir states;
            possible options are: None (no dimensionality reduction) or 'pca'
        n_dim = number of resulting dimensions after the dimensionality reduction procedure
        
    Readout parameters:
        w_ridge = regularization parameter of the ridge regression readout (only for readout_type=='lin')
    """
    
    def __init__(self,
                # reservoir
                reservoir=None,     
                n_internal_units=None,
                spectral_radius=None,
                leak=None,
                connectivity=None,
                input_scaling=None,
                noise_level=None,
                n_drop=None,
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
        """
        Train the RC model for forecasting.

        Parameters
        ----------
        X : array-like, shape [T, V]
            Training data.
        
        Y : array-like, shape [T, V]
            Target values.
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

        # ============ Train readout ============
        self.readout.fit(red_states, Y[self.n_drop:,:])          
            
        if verbose:
            tot_time = (time.time()-time_start)/60
            print(f"Training completed in {tot_time:.2f} min")
    
    
    def predict(self, Xte):
        """
        Compute predictions for test data.

        Parameters
        ----------
        Xte : array-like, shape [T, V]
            Test data.

        Returns
        -------
        Yhat : array, shape [T, V]
            Predicted values.
        """

        # ============ Compute reservoir states ============
        res_states_te = self._reservoir.get_states(Xte[None,:,:], n_drop=self.n_drop, bidir=False) 
        
        # ============ Dimensionality reduction of the reservoir states ============ 
        if self.dimred_method is not None:
            if self.dimred_method.lower() == 'pca':
                red_states_te = self._dim_red.transform(res_states_te[0])                          
        else: # Skip dimensionality reduction
            red_states_te = res_states_te[0]        
                    
        # ============ Apply readout ============
        Yhat = self.readout.predict(red_states_te)
            
        return Yhat