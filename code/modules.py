# General imports
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import maxout
import time
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tensorPCA import tensorPCA
from scipy.spatial.distance import pdist, cdist, squareform

# Custom imports
from tf_utils import train_tf_model
from reservoir import Reservoir
from kafnets import kaf


def fc_layer(input_, in_dim, size):
    """
    Define and evaluate a fully connected layer.
    
    Parameters:
        input_: TF placeholder for fc layer input
        in_dim: dimension of the input
        size: number of neurons in the layer
    """
    W = tf.Variable(
            tf.random_normal(
                shape=(in_dim, size),
                stddev=math.sqrt(4.0 / (in_dim + size)),
                ),
            name='mlp_weights'
            )

    b = tf.Variable(tf.zeros([size]), name='mlp_biases')

    with tf.name_scope('Wx_plus_b'):
        result = tf.add(tf.matmul(input_, W), b)

    return result


def build_MLP(input_, in_dim, num_layers, n_classes, keep_prob, nonlinearity='relu'):
    """
    Build a parametric MLP.
    
    Parameters:
        input_: TF placeholder for MLP input
        in_dim: dimension of the input
        num_layers: number of layers of the MLP
        n_classes: number of output classes (output dimension)
        keep_prob: TF placeholder for probability of keeping connections in dropout
        nonlinearity: type of activation function {'relu', 'tanh', 'maxout', 'kaf'}
    """
    
    for i, neurons in enumerate(num_layers):
        with tf.name_scope('h{}'.format(i)):
            if nonlinearity == 'relu':
                layer = fc_layer(input_, in_dim, neurons)
                layer = tf.nn.relu(layer)
            elif nonlinearity == 'tanh':
                layer = fc_layer(input_, in_dim, neurons)
                layer = tf.nn.tanh(layer)
            elif nonlinearity == 'maxout':
                K = 3
                layer = fc_layer(input_, in_dim, neurons * K)
                layer = maxout(layer, neurons)
            elif nonlinearity == 'kaf':
                layer = fc_layer(input_, in_dim, neurons)
                layer = kaf(layer, 'kaf_' + str(i))
            else:
                raise RuntimeError('Unknown nonlinearity, can be {relu, tanh, maxout, kaf}')
            
            layer = tf.nn.dropout(layer, keep_prob=keep_prob)

            input_ = layer
            in_dim = neurons

    with tf.name_scope('out'):
        logits = fc_layer(input_, in_dim, n_classes)

    return logits


def compute_test_scores(pred_class, Yte):
    """
    Compute classification accuracy and F1 score
    """
    
    true_class = np.argmax(Yte, axis=1)
    
    accuracy = accuracy_score(true_class, pred_class)
    if Yte.shape[1] > 2:
        f1 = f1_score(true_class, pred_class, average='weighted')
    else:
        f1 = f1_score(true_class, pred_class, average='binary')

    return accuracy, f1

def train_RC_classifier(
              X, Y,
              Xte, Yte,
              # reservoir
              reservoir=None,     
              n_internal_units=None,
              spectral_radius=None,
              connectivity=None,
              input_scaling=None,
              noise_level=None,
              n_drop=None,
              bidir=False,
              # dim red
              dimred_method=None, 
              n_dim=None,
              # representation
              mts_rep=None,
              w_ridge_embedding=None,
              # readout
              readout_type=None,               
              w_ridge=None,              
              fc_layout=None,
              batch_size=None,
              num_epochs=None,
              p_keep=None,
              w_l2=None,
              learning_rate=None,
              nonlinearity=None, 
              svm_gamma=1.0,
              svm_C=1.0,
              ):
    """
    Build and evaluate a RC-based classifier.
    The training and test MTS are multidimensional arrays of shape [N,T,V], with
        - N = number of samples
        - T = number of time steps in each sample
        - V = number of variables in each sample
    Training and test labels have shape [N,1]
    
    Dataset parameters:
        X, Y = training data and respective labels
        Xte, Yte = test data and respective labels
        
    Reservoir parameters:
        reservoir = precomputed reservoir (oject of class 'Reservoir');
            if None, the following structural hyperparameters must be specified
        n_internal_units = processing units in the reservoir
        spectral_radius = largest eigenvalue of the reservoir matrix of connection weights
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
            'mlp' (multiplayer perceptron) or 'svm'          
        w_ridge = regularization parameter of the ridge regression readout (only for readout_type=='lin')              
        fc_layout = list with the sizes of MLP layers, e.g. [20,20,10] defines a MLP with 3 layers 
            of 20, 20 and 10 units respectively (only for readout_type=='mlp')
        batch_size = size of the mini batches used during training (only for readout_type=='mlp')
        num_epochs = number of iterations during the optimization (only for readout_type=='mlp')
        p_keep = probability of keeping connections in dropout (only for readout_type=='mlp')
        w_l2 = weight of the L2 regularization (only for readout_type=='mlp')
        learning_rate = learning rate in the gradient descent optimization (only for readout_type=='mlp')
        nonlinearity = type of activation function {'relu', 'tanh', 'maxout', 'kaf'} (only for readout_type=='mlp')
        svm_gamma = bandwith of the RBF kernel (only for readout_type=='svm')
        svm_C = regularization for SVM hyperplane (only for readout_type=='svm')
    """
    time_start = time.time()
    
    # ============ Generate sequence of reservoir states ============
    
    # Initialize reservoir
    if reservoir is None:
        
        if n_internal_units is None \
        or spectral_radius is None \
        or connectivity is None \
        or input_scaling is None \
        or noise_level is None:
            raise RuntimeError('Reservoir parameters missing')
            
        else:
            reservoir = Reservoir(n_internal_units, spectral_radius, connectivity,
                                  input_scaling, noise_level)

    # Compute reservoir states
    res_states = reservoir.get_states(X, n_drop=n_drop, bidir=bidir) # train
    res_states_te = reservoir.get_states(Xte, n_drop=n_drop, bidir=bidir) # test

    # ============ Dimensionality reduction of the reservoir states ============
    
    # Reduce dimension with PCA or tensorPCA
    if dimred_method is not None:
        if dimred_method.lower() == 'pca':
            dim_red = PCA(n_components=n_dim)            
        elif dimred_method.lower() == 'tenpca':
            dim_red = tensorPCA(n_components=n_dim)
        else:
            raise RuntimeError('Invalid dimred method ID')
            
        if dimred_method.lower() == 'pca':
            # matricize
            N_samples = res_states.shape[0]
            N_samples_te = res_states_te.shape[0]
            res_states = res_states.reshape(-1, res_states.shape[2])
            res_states_te = res_states_te.reshape(-1, res_states_te.shape[2])                    
            # ..transform..
            red_states = dim_red.fit_transform(res_states)
            red_states_te = dim_red.transform(res_states_te)            
            # ..and put back in tensor form
            red_states = red_states.reshape(N_samples,-1,red_states.shape[1])
            red_states_te = red_states_te.reshape(N_samples_te,-1,red_states_te.shape[1])            
        else: # tensorPCA
            red_states = dim_red.fit_transform(res_states)
            red_states_te = dim_red.transform(res_states_te)
    
    # Skip dimensionality reduction                  
    else:
        red_states = res_states
        red_states_te = res_states_te 
    
    # ============ Generate representation of the MTS ============
    if mts_rep=='output' or mts_rep=='reservoir':
        ridge_embedding = Ridge(alpha=w_ridge_embedding, fit_intercept=True)
        coeff_tr = []
        biases_tr = []  
        coeff_te = []
        biases_te = []  
        
        # Output model space representation
        if mts_rep=='output':
            if bidir:
                X = np.concatenate((X,X[:, ::-1, :]),axis=2)
                Xte = np.concatenate((Xte,Xte[:, ::-1, :]),axis=2)  
                
            for i in range(X.shape[0]):
                ridge_embedding.fit(red_states[i, 0:-1, :], X[i, n_drop+1:, :])
                coeff_tr.append(ridge_embedding.coef_.ravel())
                biases_tr.append(ridge_embedding.intercept_.ravel())
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)

            for i in range(Xte.shape[0]):
                ridge_embedding.fit(red_states_te[i, 0:-1, :], Xte[i, n_drop+1:, :])
                coeff_te.append(ridge_embedding.coef_.ravel())
                biases_te.append(ridge_embedding.intercept_.ravel())
            input_repr_te = np.concatenate((np.vstack(coeff_te), np.vstack(biases_te)), axis=1)
        
        # Reservoir model space representation
        else:
            for i in range(X.shape[0]):
                ridge_embedding.fit(red_states[i, 0:-1, :], red_states[i, 1:, :])
                coeff_tr.append(ridge_embedding.coef_.ravel())
                biases_tr.append(ridge_embedding.intercept_.ravel())
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)

            for i in range(Xte.shape[0]):
                ridge_embedding.fit(red_states_te[i, 0:-1, :], red_states_te[i, 1:, :])
                coeff_te.append(ridge_embedding.coef_.ravel())
                biases_te.append(ridge_embedding.intercept_.ravel())
            input_repr_te = np.concatenate((np.vstack(coeff_te), np.vstack(biases_te)), axis=1)
    
    # Last state representation        
    elif mts_rep=='last':
        input_repr = red_states[:, -1, :]
        input_repr_te = red_states_te[:, -1, :]
    else:
        raise RuntimeError('Invalid representation ID')
                           
    # ============ Readout definition, training & test ============
    
    # Ridge regression
    if readout_type == 'lin':
        readout = Ridge(alpha=w_ridge)
        readout.fit(input_repr, Y)          
        logits = readout.predict(input_repr_te)
        pred_class = np.argmax(logits, axis=1)
    
    # MLP (deep readout)    
    elif readout_type == 'mlp':
        num_classes = Y.shape[1] 
        input_size = input_repr.shape[1]
        with tf.Graph().as_default():
    
            # tf Graph input
            nn_input = tf.placeholder(shape=(None, input_size),dtype=tf.float32,name='nn_input')
            nn_output = tf.placeholder(shape=(None, num_classes),dtype=tf.float32,name='nn_output')
            keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')
    
            # Instantiate MLP readout
            logits = build_MLP(
                    nn_input,
                    input_size,
                    fc_layout,
                    num_classes,
                    keep_prob,
                    nonlinearity=nonlinearity)
    
            # Train MLP and evaluate it on the test
            pred_class = train_tf_model('ESN', input_repr, Y, input_repr_te, Yte,
                                                                batch_size, num_epochs, nn_input, keep_prob,
                                                                logits, nn_output, w_l2, learning_rate, p_keep)    
    
    # SVM readout
    elif readout_type == 'svm':
        svc = SVC(C=svm_C, kernel='precomputed')
                
        # training
        w_dist = pdist(np.vstack(coeff_tr), metric='sqeuclidean')
        b_dist = pdist(np.vstack(biases_tr), metric='sqeuclidean')
        Ktr = squareform(w_dist + b_dist) 
        Ktr = np.exp(-svm_gamma*Ktr)
        svc.fit(Ktr, np.argmax(Y,axis=1))
        
        # test
        w_dist = cdist(np.vstack(coeff_te), np.vstack(coeff_tr), metric='sqeuclidean')
        b_dist = cdist(np.vstack(biases_te), np.vstack(biases_tr), metric='sqeuclidean')
        Kte = w_dist+b_dist
        Kte = np.exp(-svm_gamma*Kte)
        pred_class = svc.predict(Kte)
        
    else:
        raise RuntimeError('Invalid reservoir type')   
        
    accuracy, f1 = compute_test_scores(pred_class, Yte)
    tot_time = (time.time()-time_start)/60

    return accuracy, f1, tot_time