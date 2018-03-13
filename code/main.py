# General imports
import numpy as np
import scipy.io
from sklearn.preprocessing import OneHotEncoder

# Custom imports
from modules import train_RC_classifier

# ============ Network configuration and hyperparameter values ============
config = {}
config['dataset_name'] = 'JpVow'

# Parameters for the reservoir
config['n_internal_units'] = 800 # size of the reservoir
config['connectivity'] = 0.25 # percentage of nonzero connections in the reservoir
config['spectral_radius'] = 0.99 # largest eigenvalue of the reservoir
config['input_scaling'] = 0.15 # scaling of the input weights
config['noise_level'] = 0.001 # noise in the reservoir state update
config['n_drop'] = 5 # transient states to be dropped
config['bidir'] = False # bidirectional or unidirectional reservoir

# Dimensionality reduction parameters
config['dimred_method'] ='tenpca'
config['n_dim'] = 75

# Type of MTS representation
config['mts_rep'] = 'reservoir'
config['w_ridge_embedding'] = 5.0

# Type of readout
config['readout_type'] = 'lin'

# Linear readout parameters
config['w_ridge'] = 1.0

# MLP readout parameters
config['mlp_layout'] = [20,20]
config['batch_size'] = 25 # samples in the mini-batches in gradient descent training
config['num_epochs'] = 2000 # number of epochs 
config['p_keep'] = 1.0
config['w_l2'] = 0.0001
config['learning_rate'] = 0.001
config['nonlinearity'] = 'maxout'

# SVM readout
config['svm_gamma'] = 1.0
config['svm_C'] = 1.0

print(config)

# ============ Load dataset ============
data = scipy.io.loadmat('../dataset/'+config['dataset_name']+'.mat')
X = data['X']  # shape is [N,T,V]
if len(X.shape) < 3:
    X = np.atleast_3d(X)
Y = data['Y']  # shape is [N,1]
Xte = data['Xte']
if len(Xte.shape) < 3:
    Xte = np.atleast_3d(Xte)
Yte = data['Yte']

print('Loading '+config['dataset_name']+' - Tr: '+ str(X.shape)+', Te: '+str(Xte.shape))

# One-hot encoding for labels
onehot_encoder = OneHotEncoder(sparse=False)
Y = onehot_encoder.fit_transform(Y)
Yte = onehot_encoder.transform(Yte)

# ============ Train and evaluate model ============
accuracy, f1, tot_time =  train_RC_classifier(X=X,
                                              Y=Y,
                                              Xte=Xte,
                                              Yte=Yte,
                                              reservoir=None,     
                                              n_internal_units=config['n_internal_units'],
                                              spectral_radius=config['spectral_radius'],
                                              connectivity=config['connectivity'],
                                              input_scaling=config['input_scaling'],
                                              noise_level=config['noise_level'],
                                              n_drop=config['n_drop'],
                                              bidir=config['bidir'],
                                              dimred_method=config['dimred_method'], 
                                              n_dim=config['n_dim'],
                                              mts_rep=config['mts_rep'],
                                              w_ridge_embedding=config['w_ridge_embedding'],
                                              readout_type=config['readout_type'],            
                                              w_ridge=config['w_ridge'],              
                                              fc_layout=config['mlp_layout'],
                                              batch_size=config['batch_size'],
                                              num_epochs=config['num_epochs'],
                                              p_keep=config['p_keep'],
                                              w_l2=config['w_l2'],
                                              learning_rate=config['learning_rate'],
                                              nonlinearity=config['nonlinearity'], 
                                              svm_gamma=config['svm_gamma'],
                                              svm_C=config['svm_C'])

print('Accuracy = %.3f, F1 = %.3f, Tot time = %.2f'%(accuracy, f1, tot_time))
        

