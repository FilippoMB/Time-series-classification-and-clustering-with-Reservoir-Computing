import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import v_measure_score

from reservoir_computing.modules import RC_model
from reservoir_computing.datasets import ClfLoader

np.random.seed(0) # Fix the seed for reproducibility

# Load the data
Xtr, Ytr, Xte, Yte = ClfLoader().get_data('Japanese_Vowels')

# Since we are doing clustering, we do not need the train/test split
X = np.concatenate((Xtr, Xte), axis=0)
Y = np.concatenate((Ytr, Yte), axis=0)

# Initialize the RC model
rcm =  RC_model(n_internal_units=400,
                readout_type=None # by setting None, the input representations will be stored
                )

# Generate representations of the input MTS
rcm.fit(X)
mts_representations = rcm.input_repr

# Compute Dissimilarity matrix
Dist = cosine_distances(mts_representations)
distArray = ssd.squareform(Dist)

# Hierarchical clustering
distArray = ssd.squareform(Dist)
Z = linkage(distArray, 'ward')
clust = fcluster(Z, t=4.0, criterion="distance")
print(f"Found {len(np.unique(clust))} clusters")

# Evaluate the agreement between class and cluster labels
nmi = v_measure_score(Y[:,0], clust)
print(f"Normalized Mutual Information (v-score): {nmi:.3f}")