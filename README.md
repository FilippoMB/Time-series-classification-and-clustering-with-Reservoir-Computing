# Reservoir-model-space-classifier

<img src="./logs/RC_classifier.JPG" width="603.5" height="272.5">

With this library is possible to quickly implement a classifier for time series data based on Reservoir Computing, the family of approaches popularized in machine learning by Echo State Networks.

Several options are available to customize the RC classifier: (i) specify the reservoir configuration, (ii) perform a dimensionality reduction on the produced sequence of the reservoir's states, (iii) generate a representation of the time series from the sequence of reservoir's states, and (iv) perform the final classification -- see the figure above. 
The library allows to define the *reservoir model space* as representation for the time series: details of the methodology can be found in the [original paper](https://arxiv.org/abs/1803.07870).

**Quick execution**

Run the script ```example.py``` to perform a quick execution of the RC classifier on a benchmark dataset for classification of multivariate time series.
The code has been tested on Python 3.5.

Required libraries:

- Tensorflow (tested on version 1.8.0)
- sklearn (tested on version 0.19.1)
- scipy


## Configure the RC-classifier

The main class ```RC_classifier``` contained in [modules.py](https://github.com/FilippoMB/Reservoir-model-space-classifier/blob/master/code/modules.py) permits to specify, train and test an RC-classifier.
The RC-classifier is configured by passing to the constructor of the class ```RC_classifier``` a set of parameters. To get an idea, you can check in ```example.py``` where the parameters are specified through a dictionary (config). 

The available configuration parameters are listed in the following and, for the sake of clarity, are grouped according to which part of the architecture they refer to.

    
**1. Reservoir parameters:**

- n_drop - number of transient states to drop
- bidir - use a bidirectional reservoir (True or False)
- reservoir - precomputed reservoir (oject of class ````Reservoir```` in [reservoir.py](https://github.com/FilippoMB/Reservoir-model-space-classifier/blob/master/code/reservoir.py); if ```None```, the following structural hyperparameters must be specified:
    - n\_internal\_units = number of processing units in the reservoir
    - spectral_radius = largest eigenvalue of the reservoir matrix of connection weights (to guarantee the Echo State Property, set spectral\_radius <= leak <= 1)
    - leak = amount of leakage in the reservoir state update (optional, None or 1.0 --> no leakage)
    - circ = if True, generate a determinisitc reservoir with circle topology where each connection has the same weight
    - connectivity = percentage of nonzero connection weights (ignored if circ = True)
    - input_scaling = scaling of the input connection weights (note that weights are randomly drawn from {-1,1})
    - noise_level = deviation of the Gaussian noise injected in the state update
            
**2. Dimensionality reduction parameters:**

- dimred_method - procedure for reducing the number of features in the sequence of reservoir states; possible options are: ````None```` (no dimensionality reduction), ````'pca'```` (standard PCA) or ````'tenpca'```` (tensorial PCA for multivariate time series data)
- n_dim - number of resulting dimensions after the dimensionality reduction procedure
    
**3. Representation parameters:**

- mts_rep - type of multivariate time series representation. It can be ````'last'```` (last state), ````'output'```` (output model space), or ````'reservoir'```` (reservoir model space)
- w\_ridge\_embedding - regularization parameter of the ridge regression in the output model space and reservoir model space representation; ignored if mts_rep is ````None````
    
**4. Readout parameters:**

- readout_type - type of readout used for classification. It can be ````'lin'```` (ridge regression), ````'mlp'```` (multilayer perceptron) or ````'svm'````          
- w\_ridge - regularization parameter of the ridge regression readout (only when readout_type is ````'lin'````)              
- mlp\_layout - list with the sizes of MLP layers, e.g. ````[20,20,10]```` defines a MLP with 3 layers of 20, 20 and 10 units respectively (only when readout_type is ````'mlp'````)
- batch\_size - size of the mini batches used during training (only when readout_type is ````'mlp'````)
- num\_epochs - number of iterations during the optimization (only when readout_type is ````'mlp'````)
- p\_drop - probability of dropping connections in dropout (only when readout_type is ````'mlp'````)
- w\_l2 = weight of the L2 regularization (only when readout_type is ````'mlp'````)
- learning\_rate = learning rate in the gradient descent optimization (only when readout_type is ````'mlp'````)
- nonlinearity = type of activation function; it can be ````{'relu', 'tanh', 'sigmoid', 'lin', 'maxout', 'kaf'}```` (only when readout_type is ````'mlp'````)
- svm\_gamma = bandwith of the RBF kernel (only when readout_type is ````'svm'````)
- svm\_C = regularization for the SVM hyperplane (only when readout_type is ````'svm'````)

## Train and test the RC-classifier

The training and test function requires in input training and test data, which must be provided as multidimensional NumPy arrays of shape *[N,T,V]*, with:

- *N* = number of samples
- *T* = number of time steps in each sample
- *V* = number of variables in each sample

Training and test labels (Y and Yte) must be provided in one-hot encoding format, i.e. a matrix *[N,C]*, where *C* is the number of classes.

**Training**

````RC_classifier.train````

Inputs:

- X, Y: training data and respective labels

Outputs:

- tr\_time: time (in seconds) used to train the classifier

**Test**

````RC_classifier.test````

Inputs:

- Xte, Yte: test data and respective labels

Outputs:

- accuracy, F1 score: metrics achieved on the test data