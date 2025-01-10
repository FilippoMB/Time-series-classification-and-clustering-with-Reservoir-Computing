# Time series classification and clustering with Reservoir Computing


[![arXiv](https://img.shields.io/badge/arXiv-1803.07870-b31b1b.svg)](https://arxiv.org/abs/1803.07870)
[![Downloads](https://static.pepy.tech/badge/reservoir-computing)](https://pepy.tech/project/reservoir-computing)

<div align="center" id="fig:RCClassifier">
<img src="https://github.com/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/raw/master/docs/RC_classifier.png" style="width: 22cm">
<p>
Figure 1: Overview of the RC classifier.
</div>

This library allows to quickly implement different architectures for time series data based on *Reservoir Computing (RC)*, the family of approaches popularized in machine learning by Echo State Networks.
This library is primarly design to perform **classification** and **clustering** of both univariate and multivariate time series. However, it can also be used to perform time series **forecasting**. 

<div align="center">
📚 <a href="https://reservoir-computing.readthedocs.io/en/latest">Documentation</a> - 🚀 <a href="https://reservoir-computing.readthedocs.io/en/latest/index.html#quick-start">Getting Started</a> - 🧩 <a href="https://reservoir-computing.readthedocs.io/en/latest/index.html#advanced-examples">Advanced examples</a>
</div>

# 🛠️ Setup

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

# 🚀 Getting Started

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

[![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.org/github/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/blob/master/examples/notebooks/classification.ipynb) 
or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xk7ZiT3yGrpFrsmL4CR0Q31REr6K_soA?usp=sharing)

**Clustering**

````bash
python examples/clustering_example.py
````

[![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.org/github/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/blob/master/examples/notebooks/clustering.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ycpnifi6yCRZC8_N15CiltwzSRzJTnVN?usp=sharing)

**Forecasting**

````bash
python examples/forecasting_example.py
````
[![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.org/github/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/blob/master/examples/notebooks/forecasting.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fnt3IMmlehD9-P7hNVuiVDH_saeIOayk?usp=sharing)

# 👀 Overview of the framework

In the following, we present the three main functionalities of this library.

## Classification

Referring to [Figure 1](#fig:RCClassifier), the RC classifier consists of four different modules.

1. The **reservoir** module specifies the reservoir configuration (*e.g.*, bidirectional, leaky neurons, circle topology). Given a multivariate time series $\mathbf{X}$ it generates a sequence of the same length of Reservoir states $\mathbf{H}$.
2. The **dimensionality reduction** module (optionally) applies a dimensionality reduction on the  sequence of the reservoir's states $\mathbf{H}$ generating a new sequence $\mathbf{\bar H}$.
3. The **representation** generates a vector $\mathbf{r}_\mathbf{X}$ from the sequence of reservoir's states, which represents in vector form the original time series $\mathbf{X}$.
4. The **readout** module is a classifier that maps the representation $\mathbf{r}_\mathbf{X}$ into the class label $\mathbf{y}$, associated with the time series $\mathbf{X}$. 

> [!Note] 
> This library implements also the *reservoir model space*, a very powerful representation $\mathbf{r}_\mathbf{X}$ for the time series. Details about the methodology are found in the [original paper](https://arxiv.org/abs/1803.07870).

The class `RC_model` contained in [modules.py](https://github.com/FilippoMB/Reservoir-model-space-classifier/blob/master/code/modules.py) permits to specify, train and test an RC-model.
Several options are available to customize the RC model, by selecting different configurations for each module.

The training and test function requires in input training and test data, which must be provided as multidimensional numpy arrays of shape *[N,T,V]*, with:

- *N* = number of samples
- *T* = number of time steps in each sample
- *V* = number of variables in each sample

Training and test labels (`Ytr` and `Yte`) must be provided in one-hot encoding format, i.e. a matrix *[N,C]*, where *C* is the number of classes.

````python
from reservoir_computing.modules import RC_model

clf = RC_model()
clf.fit(Xtr, Ytr) # Training
Yhat = clf.predict(Xte) # Prediction
````

## Clustering
The representation $\mathbf{r}_\mathbf{X}$ obtained from the **representation** module (step 3) can be used to perform time series clustering.
The same class `RC_model` used for classification can be configured to directly return the time series representations, which can be used in unsupervised tasks such as clustering and dimensionality reduction.

As in the case of classification, the data must be provided as multidimensional NumPy arrays of shape *[N,T,V]*

````python
from reservoir_computing.modules import RC_model

clst = RC_model(readout_type=None)
clst.fit(X)
rX = clst.input_repr # representations of the input data
````

The representations `rX` can be used to perfrom clustering using traditional clustering algorithms for vectorial data, such as those from [sk-learn](https://scikit-learn.org/stable/modules/clustering.html).

## Forecasting
The sequences $\mathbf{H}$ and $\mathbf{\bar H}$ obtained at steps 1 and 2 can be directly used to forecast the future values of the time series.

The class `RC_forecaster` contained in [modules.py](https://github.com/FilippoMB/Reservoir-model-space-classifier/blob/master/code/modules.py) permits to specify, train and test an RC-model for time series forecasting.


````python
from reservoir_computing.modules import RC_forecaster

fcst = RC_forecaster()
fcst.fit(Xtr, Ytr) # Training
Yhat = fcst.predict(Xte) # Predictions
````

Here, `Xtr`, `Ytr` are current and future values, respectively, used for training.

# 🧩 Advanced examples
The following notebooks illustrate more advanced use-cases.

- Perform dimensionality reduction, cluster analysis, and visualize the results: [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.org/github/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/blob/master/examples/notebooks/clustering_visualization.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N19l9hH7cxFdWxeAuL7WFbxZFdSIkl-D?usp=sharing)
- Probabilistic forecasting with advanced regression models as readout: [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.org/github/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/blob/master/examples/notebooks/forecasting_with_GBRT.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HAD2IlaOggaRGIuIPCabUje2IClqzpyn?usp=sharing)
- Use advanced classifiers as readout: [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.org/github/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/blob/master/examples/notebooks/advanced_classifiers.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RCpDpAxuYFuFuoRWSM7iMA1sRx8exlhH?usp=sharing)
- Impute missing data in time series [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/blob/master/examples/notebooks/missing_data_imputation.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/blob/master/examples/notebooks/missing_data_imputation.ipynb)
- Reconstruct the attractor of a dynamical system in the phase space [![nbviewer](https://img.shields.io/badge/-View-blue?logo=jupyter&style=flat&labelColor=gray)](https://nbviewer.jupyter.org/github/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/blob/master/examples/notebooks/state_space_reconstruction.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/blob/master/examples/notebooks/state_space_reconstruction.ipynb)

# 📦 Datasets

There are several datasets available to perform time series classification/clustering and forecasting.

**Classification and clustering**

````python
from reservoir_computing.datasets import ClfLoader

downloader = ClfLoader()
downloader.available_datasets(details=True)  # Print available datasets
Xtr, Ytr, Xte, Yte = downloader.get_data('Libras')  # Download dataset and return data
````

**Forecasting**

Real-world time series

````python
from reservoir_computing.datasets import PredLoader

downloader = PredLoader()
downloader.available_datasets(details=False)  # Print available datasets
X = downloader.get_data('CDR')  # Download dataset and return data
````

Synthetic time series

````python
from reservoir_computing.datasets import SynthLoader

synth = SynthLoader()
synth.available_datasets()  # Print available datasets
Xs = synth.get_data('Lorenz')  # Generate synthetic time series
````

# 📝 Citation

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

# 🌐 License
The code is released under the MIT License. See the attached LICENSE file.