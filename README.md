[![arXiv](https://img.shields.io/badge/arXiv-1803.07870-b31b1b.svg)](https://arxiv.org/abs/1803.07870)
[![Downloads](https://static.pepy.tech/badge/reservoir-computing)](https://pepy.tech/project/reservoir-computing)

# Framework overview

<img src="https://github.com/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/raw/master/docs/source/_static/img/RC_classifier.JPG" width="603.5" height="320">

This library allows to quickly implement different architectures for time series data based on *Reservoir Computing (RC)*, the family of approaches popularized in machine learning by Echo State Networks.
This library is primarly design to perform **classification** and **clustering** of both univariate and multivariate time series. However, it can also be used to perform time series **forecasting**. 

### Classification
Several options are available to customize the RC model, by selecting different configurations for each module.
1. The **reservoir** module specifies the reservoir configuration (*e.g.*, bidirectional, leaky neurons, circle topology). Given a multivariate time series $\mathbf{X}$ it generates a sequence of the same length of Reservoir states $\mathbf{H}$.
2. The **dimensionality reduction** module (optionally) applies a dimensionality reduction on the  sequence of the reservoir's states $\mathbf{H}$ generating a new sequence $\mathbf{\bar H}$.
3. The **representation** generates a vector $\mathbf{r}_\mathbf{X}$ from the sequence of reservoir's states, which represents in vector form the original time series $\mathbf{X}$.
4. The **readout** module is a classifier that maps the representation $\mathbf{r}_\mathbf{X}$ into the class label $\mathbf{y}$, associated with the time series $\mathbf{X}$. 

This library implements the *reservoir model space* a very powerful representation $\mathbf{r}_\mathbf{X}$ for the time series.
Details about the methodology are found in the [original paper](https://arxiv.org/abs/1803.07870).

### Clustering
The representation $\mathbf{r}_\mathbf{X}$ obtained at step 3 can be used to perform time series clustering.

### Forecasting
The sequences $\mathbf{H}$ and $\mathbf{\bar H}$ obtained at steps 1 and 2 can be directly used to forecast the future values of the time series.

## Installation

The recommended installation is with pip:

````bash
pip install reservoir-computing
````

Alternatively, you can install the library from source:
````bash
git clone https://github.com/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing.git
cd Time-series-classification-and-clustering-with-Reservoir-Computing
pip install -e .
````

## Quick start

The following scripts provide minimalistic examples that illustrate how to use the library for different tasks.

To run them, download the project and cd to the root folder:

````bash
git clone https://github.com/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing.git
cd Time-series-classification-and-clustering-with-Reservoir-Computing
````

**Classification**

````bash
python examples/classification_example.py
````

**Clustering**

````bash
python examples/clustering_example.py
````

**Forecasting**

````bash
python examples/forecasting_example.py
````

The following notebooks illustrate more advanced use-cases.

- Perform time series cluster analysis and visualize the results: [view](https://nbviewer.org/github/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/blob/master/notebooks/clustering_visualization.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_BahDTpTB8dEHARJtCLNe0JxVT8WAXuD?usp=sharing)
- Probabilistic forecasting with advanced regression models as readout: [view](https://nbviewer.org/github/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/blob/master/notebooks/prediction_with_GBRT.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wxIRcX_y7572x6WeezfWfJiTurVsc2aW?usp=sharing)
- Use advanced classifiers as readout: [view](https://nbviewer.org/github/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/blob/master/notebooks/advanced_classifiers.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PnH_lffbE1nfghYwPa2j0H2_4mgtRwQo?usp=sharing)

## Configure the RC-model

The main class ```RC_model``` contained in [modules.py](https://github.com/FilippoMB/Reservoir-model-space-classifier/blob/master/code/modules.py) permits to specify, train and test an RC-model.
The RC-model is configured by passing to the constructor of the class ```RC_model``` a set of parameters. To get an idea, you can check ```classification_example.py``` or ```clustering_example.py``` where the parameters are specified through a dictionary (````config````). 

The available configuration hyperparameters are listed in the following and, for the sake of clarity, are grouped according to which module of the architecture they refer to.

    
**1. Reservoir:**

- n_drop - number of transient states to drop
- bidir - use a bidirectional reservoir (True or False)
- reservoir - precomputed reservoir (object of class ````Reservoir```` in [reservoir.py](https://github.com/FilippoMB/Reservoir-model-space-classifier/blob/master/code/reservoir.py); if ```None```, the following hyperparameters must be specified:
    - n\_internal\_units = number of processing units in the reservoir
    - spectral_radius = largest eigenvalue of the reservoir matrix of connection weights (to guarantee the Echo State Property, set spectral\_radius <= leak <= 1)
    - leak = amount of leakage in the reservoir state update (optional, ````None```` or ````1.0```` --> no leakage)
    - circ = if True, generate a determinisitc reservoir with circle topology where each connection has the same weight
    - connectivity = percentage of nonzero connection weights (ignored if circ = ````True````)
    - input_scaling = scaling of the input connection weights (note that weights are randomly drawn from {-1,1})
    - noise_level = deviation of the Gaussian noise injected in the state update
            
**2. Dimensionality reduction:**

- dimred_method - procedure for reducing the number of features in the sequence of reservoir states; possible options are: ````None```` (no dimensionality reduction), ````'pca'```` (standard PCA) or ````'tenpca'```` (tensorial PCA for multivariate time series data)
- n_dim - number of resulting dimensions after the dimensionality reduction procedure
    
**3. Representation:**

- mts_rep - type of multivariate time series representation. It can be ````'last'```` (last state), ````'mean'```` (mean of all states), ````'output'```` (output model space), or ````'reservoir'```` (reservoir model space)
- w\_ridge\_embedding - regularization parameter of the ridge regression in the output model space and reservoir model space representation; ignored if mts_rep is ````None````
    
**4. Readout:**

- readout_type - type of readout used for classification. It can be ````'lin'```` (ridge regression), ````'mlp'```` (multilayer perceptron), ````'svm'```` (support vector machine), or ````None````. If ````None````, the input representations will be stored in the ````.input_repr```` attribute: this is useful for clustering and visualization. Also, if ````None````, the other Readout hyperparameters can be left unspecified.
- w\_ridge - regularization parameter of the ridge regression readout (only when readout_type is ````'lin'````)              
- mlp\_layout - list with the sizes of MLP layers, e.g. ````[20,20,10]```` defines a MLP with 3 layers of 20, 20 and 10 units respectively (only when readout_type is ````'mlp'````)
- batch\_size - size of the mini batches used during training (only when readout_type is ````'mlp'````)
- num\_epochs - number of iterations during the optimization (only when readout_type is ````'mlp'````)
- w\_l2 = weight of the L2 regularization (only when readout_type is ````'mlp'````)
- learning\_rate = learning rate in the gradient descent optimization (only when readout_type is ````'mlp'````)
- nonlinearity = type of activation function; it can be ````{'relu', 'tanh', 'logistic', 'identity'}```` (only when readout_type is ````'mlp'````)
- svm\_gamma = bandwith of the RBF kernel (only when readout_type is ````'svm'````)
- svm\_C = regularization for the SVM hyperplane (only when readout_type is ````'svm'````)

## RC-model for classification

The training and test function requires in input training and test data, which must be provided as multidimensional NumPy arrays of shape *[N,T,V]*, with:

- *N* = number of samples
- *T* = number of time steps in each sample
- *V* = number of variables in each sample

Training and test labels (Y and Yte) must be provided in one-hot encoding format, i.e. a matrix *[N,C]*, where *C* is the number of classes.

**Training**

````python
from reservoir_computing.modules import RC_model
clf = RC_model()
clf.fit(Xtr, Ytr)
````

Inputs:

- `Xtr`, `Ytr`: training data and labels.

Outputs:

- None

**Prediction of new samples**

````python
Yhat = clf.predict(Xte)
````

Inputs:

- `Xte`: test data.

Outputs:

- `Yhat`: prediction of the labels for the test data.


## RC-model for clustering

As in the case of classification, the data must be provided as multidimensional NumPy arrays of shape *[N,T,V]*

**Training**

````python
from reservoir_computing.modules import RC_model
clst = RC_model(readout_type=None)
clst.fit(X)
rX = clst.input_repr # representations of the input data
````

Inputs:

- `X`: time series data

Outputs:

- None

The representations `rX` can be used to perfrom clustering using traditional clustering algorithms for vectorial data, such as those [here](https://scikit-learn.org/stable/modules/clustering.html).


## RC-model for forecasting

**Training**

````python
from reservoir_computing.modules import RC_forecaster
fcst = RC_forecaster()
fcst.fit(Xtr, Ytr)
````

Inputs:

- `Xtr`, `Ytr`: current and future values used for training.

Outputs:

- None

**Predicting new data**

````python
Yhat = fcst.predict(Xte)
````

Inputs:

- `Xte`: test data.

Outputs:

- `Yhat`: forecast of the test data.

# Time series datasets for classification and clustering

- A collection of univariate and multivariate time series dataset is available for download [here](https://mega.nz/#!aZkBwYDa!JZb99GQoUn4EoJYceCK3Ihe04hhYZYuIWn018gcQM8k). 
- The dataset are provided both in MATLAB and Python (Numpy) format. 
- The original raw data come from [UCI](https://archive.ics.uci.edu/ml/index.php), [UEA](https://www.groundai.com/project/the-uea-multivariate-time-series-classification-archive-2018/), and [UCR](https://www.cs.ucr.edu/~eamonn/time_series_data/) public repositories.


# Citation

Please, consider citing the original paper if you are using this library in your reasearch

````bibtex
@article{bianchi2020reservoir,
  title={Reservoir computing approaches for representation and classification of multivariate time series},
  author={Bianchi, Filippo Maria and Scardapane, Simone and L{\o}kse, Sigurd and Jenssen, Robert},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2020},
  publisher={IEEE}
}
````
    
# License
The code is released under the MIT License. See the attached LICENSE file.
