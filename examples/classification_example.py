import requests
from io import BytesIO
import pprint
import numpy as np
import scipy.io
from sklearn.preprocessing import OneHotEncoder

from reservoir_computing.modules import RC_model
from reservoir_computing.utils import compute_test_scores

np.random.seed(0) # Fix the seed for reproducibility

# ============ RC model configuration and hyperparameter values ============
config = {}

# Hyperarameters of the reservoir
config['n_internal_units'] = 450        # size of the reservoir
config['spectral_radius'] = 0.59        # largest eigenvalue of the reservoir
config['leak'] = 0.6                    # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
config['connectivity'] = 0.25           # percentage of nonzero connections in the reservoir
config['input_scaling'] = 0.1           # scaling of the input weights
config['noise_level'] = 0.01            # noise in the reservoir state update
config['n_drop'] = 5                    # transient states to be dropped
config['bidir'] = True                  # if True, use bidirectional reservoir
config['circ'] = False                  # use reservoir with circle topology

# Dimensionality reduction hyperparameters
config['dimred_method'] = 'tenpca'      # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
config['n_dim'] = 75                    # number of resulting dimensions after the dimensionality reduction procedure

# Type of MTS representation
config['mts_rep'] = 'reservoir'         # MTS representation:  {'last', 'mean', 'output', 'reservoir'}
config['w_ridge_embedding'] = 10.0      # regularization parameter of the ridge regression

# Type of readout
config['readout_type'] = 'lin'          # readout used for classification: {'lin', 'mlp', 'svm'}

# Linear readout hyperparameters
config['w_ridge'] = 5.0                 # regularization of the ridge regression readout

# SVM readout hyperparameters
config['svm_gamma'] = 0.005             # bandwith of the RBF kernel
config['svm_C'] = 5.0                   # regularization for SVM hyperplane

# MLP readout hyperparameters
config['mlp_layout'] = (10,10)          # neurons in each MLP layer
config['num_epochs'] = 2000             # number of epochs 
config['w_l2'] = 0.001                  # weight of the L2 regularization
config['nonlinearity'] = 'relu'         # type of activation function {'relu', 'tanh', 'logistic', 'identity'}

pprint.pprint(config)

# ============ Load dataset =================
data_url = 'https://raw.githubusercontent.com/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/master/dataset/JpVow.mat'
response = requests.get(data_url)
response.raise_for_status()
data = scipy.io.loadmat(BytesIO(response.content))

Xtr = data['X']  # shape is [N,T,V]
if len(Xtr.shape) < 3:
    Xtr = np.atleast_3d(Xtr)
Ytr = data['Y']  # shape is [N,1]
Xte = data['Xte']
if len(Xte.shape) < 3:
    Xte = np.atleast_3d(Xte)
Yte = data['Yte']

print(f"Loaded data from {data_url}\nData shapes:\n Tr: {Xtr.shape}\n Te: {Xte.shape}")

# One-hot encoding for labels
onehot_encoder = OneHotEncoder(sparse_output=False)
Ytr = onehot_encoder.fit_transform(Ytr)
Yte = onehot_encoder.transform(Yte)

# ============ Initialize, train and evaluate the RC model ============
classifier =  RC_model(reservoir=None,     
                       n_internal_units=config['n_internal_units'],
                       spectral_radius=config['spectral_radius'],
                       leak=config['leak'],
                       connectivity=config['connectivity'],
                       input_scaling=config['input_scaling'],
                       noise_level=config['noise_level'],
                       circle=config['circ'],
                       n_drop=config['n_drop'],
                       bidir=config['bidir'],
                       dimred_method=config['dimred_method'], 
                       n_dim=config['n_dim'],
                       mts_rep=config['mts_rep'],
                       w_ridge_embedding=config['w_ridge_embedding'],
                       readout_type=config['readout_type'],            
                       w_ridge=config['w_ridge'],              
                       mlp_layout=config['mlp_layout'],
                       num_epochs=config['num_epochs'],
                       w_l2=config['w_l2'],
                       nonlinearity=config['nonlinearity'], 
                       svm_gamma=config['svm_gamma'],
                       svm_C=config['svm_C'])

# Train the model
tr_time = classifier.fit(Xtr, Ytr) 

# Compute predictions on test data
pred_class = classifier.predict(Xte) 
accuracy, f1 = compute_test_scores(pred_class, Yte)
print(f"Accuracy = {accuracy:.3f}, F1 = {f1:.3f}")