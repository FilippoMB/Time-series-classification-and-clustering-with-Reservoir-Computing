import requests
from io import BytesIO
import pprint
import numpy as np
import scipy.io
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import v_measure_score

from reservoir_computing.modules import RC_model

np.random.seed(0) # Fix the seed for reproducibility


# ============ RC model configuration and hyperparameter values ============
config = {}

# Reservoir
config['n_internal_units'] = 450        # size of the reservoir
config['spectral_radius'] = 0.9         # largest eigenvalue of the reservoir
config['leak'] = None                   # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
config['connectivity'] = 0.25           # percentage of nonzero connections in the reservoir
config['input_scaling'] = 0.1           # scaling of the input weights
config['noise_level'] = 0.0             # noise in the reservoir state update
config['n_drop'] = 5                    # transient states to be dropped
config['bidir'] = True                  # if True, use bidirectional reservoir
config['circ'] = False                  # use reservoir with circle topology

# Dimensionality reduction
config['dimred_method'] ='tenpca'       # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
config['n_dim'] = 75                    # number of resulting dimensions after the dimensionality reduction procedure

# MTS representation
config['mts_rep'] = 'reservoir'         # MTS representation:  {'last', 'mean', 'output', 'reservoir'}
config['w_ridge_embedding'] = 5.0       # regularization parameter of the ridge regression

# Readout
config['readout_type'] = None           # by setting None, the input representations will be stored

pprint.pprint(config)

# ============ Load dataset ============
data_url = 'https://raw.githubusercontent.com/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/master/dataset/JpVow.mat'
response = requests.get(data_url)
response.raise_for_status()
data = scipy.io.loadmat(BytesIO(response.content))
X = data['X']  # shape is [N,T,V]
if len(X.shape) < 3:
    X = np.atleast_3d(X)
Y = data['Y']  # shape is [N,1]
Xte = data['Xte']
if len(Xte.shape) < 3:
    Xte = np.atleast_3d(Xte)
Yte = data['Yte']

# Since we are doing clustering, we do not need the train/test split
X = np.concatenate((X, Xte), axis=0)
Y = np.concatenate((Y, Yte), axis=0)

print(f"Loaded data from {data_url}\nData shapes:\n X: {X.shape}")

# ============ Initialize and fit the RC model ============
rcm =  RC_model(reservoir=None,     
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
                readout_type=config['readout_type'])

# Generate representations of the input MTS
rcm.fit(X)
mts_representations = rcm.input_repr

# Compute a similarity matrix from the cosine similarity of the representations
similarity_matrix = cosine_similarity(mts_representations)
        
# Normalize the similarity matrix to the range [0,1]
similarity_matrix = (similarity_matrix + 1.0)/2.0
print(f"Similarity max: {similarity_matrix.max():.2f}, min: {similarity_matrix.min():.2f}")

# Compute Dissimilarity matrix
Dist = 1.0 - similarity_matrix
np.fill_diagonal(Dist, 0) # due to numerical errors, the diagonal might not be 0

# Hierarchical clustering
distArray = ssd.squareform(Dist)
Z = linkage(distArray, 'ward')
clust = fcluster(Z, t=2.0, criterion="distance")
print(f"Found {len(np.unique(clust))} clusters")

# Evaluate the agreement between class and cluster labels
nmi = v_measure_score(Y[:,0], clust)
print(f"Normalized Mutual Information (v-score): {nmi:.3f}")