# Reservoir-model-space-classifier

<img src="./logs/RC_classifier.JPG" width="603.5" height="272.5">

Implementation of the classifier based on the reservoir model space representation. Details of the methodology can be found in the [original paper](https://arxiv.org/abs/1711.06509).


**Quick execution**

Run the script ```main.py``` to perform a quick execution of the classifier on a benchmark dataset for classification of multivariate time series.
The code has been tested on Python 3.5.

Required libraries:

- Tensorflow (tested on version 1.5.0)
- sklearn (tested on version 0.19.1)
- scipy


## Usage instructions

The main function ```train_RC_classifier()``` contained in [modules.py](https://github.com/FilippoMB/Reservoir-model-space-classifier/blob/master/code/modules.py) builds and evaluates a RC-based classifier.

The function requires in input training and test data, which must be provided as multidimensional NumPy arrays of shape *[N,T,V]*, with:

- *N* = number of samples
- *T* = number of time steps in each sample
- *V* = number of variables in each sample

Training and test labels must be provided in one-hot encoding *[N,C]*, where *C* is the number of classes.

Different configurations of the RC classifer are obtained by calling ```train_RC_classifier()``` using different configurations (see the figure above for a schematic summary). All the available parameters are listed in the following.

**Dataset parameters:**

- X, Y - training data and respective labels
- Xte, Yte - test data and respective labels
    
**Reservoir parameters:**

- n_drop - number of transient states to drop
- bidir - use a bidirectional reservoir (True or False)
- reservoir - precomputed reservoir (oject of class ````Reservoir```` in [reservoir.py](https://github.com/FilippoMB/Reservoir-model-space-classifier/blob/master/code/reservoir.py); if ```None```, the following structural hyperparameters must be specified
    - n\_internal\_units = processing units in the reservoir
    - spectral_radius = largest eigenvalue of the reservoir matrix of connection weights
    - connectivity = percentage of nonzero connection weights
    - input_scaling = scaling of the input connection weights
    - noise_level = deviation of the Gaussian noise injected in the state update

        
**Dimensionality reduction parameters:**

- dimred_method - procedure for reducing the number of features in the sequence of reservoir states; possible options are: ````None```` (no dimensionality reduction), ````'pca'```` (standard PCA) or ````'tenpca'```` (tensorial PCA)
- n_dim - number of resulting dimensions after the dimensionality reduction procedure
    
**Representation parameters:**

- mts_rep - type of multivariate time series representation. It can be ````'last'```` (last state), ````'output'```` (output model space), or ````'reservoir'```` (reservoir model space)
- w_ridge_embedding - regularization parameter of the ridge regression in the output model space and reservoir model space representation; ignored if mts_rep is ````None````
    
**Readout parameters:**

- readout_type - type of readout used for classification. It can be ````'lin'```` (ridge regression), ````'mlp'```` (multilayer perceptron) or ````'svm'````          
- w\_ridge - regularization parameter of the ridge regression readout (only when readout_type is ````'lin'````)              
- fc\_layout - list with the sizes of MLP layers, e.g. ````[20,20,10]```` defines a MLP with 3 layers of 20, 20 and 10 units respectively (only when readout_type is ````'mlp'````)
- batch\_size - size of the mini batches used during training (only when readout_type is ````'mlp'````)
- num\_epochs - number of iterations during the optimization (only when readout_type is ````'mlp'````)
- p\_keep - probability of keeping connections in dropout (only when readout_type is ````'mlp'````)
- w\_l2 = weight of the L2 regularization (only when readout_type is ````'mlp'````)
- learning\_rate = learning rate in the gradient descent optimization (only when readout_type is ````'mlp'````)
- nonlinearity = type of activation function; it can be ````{'relu', 'tanh', 'maxout', 'kaf'}```` (only when readout_type is ````'mlp'````)
- svm\_gamma = bandwith of the RBF kernel (only when readout_type is ````'svm'````)
- svm\_C = regularization for the SVM hyperplane (only when readout_type is ````'svm'````)
