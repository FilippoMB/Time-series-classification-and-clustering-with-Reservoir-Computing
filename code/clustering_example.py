# General imports
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import scipy.spatial.distance as ssd
import matplotlib as mpl
from matplotlib.pyplot import cm
from scipy.cluster import hierarchy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import v_measure_score

# Custom imports
from modules import RC_model

# Set the colormap for the histogram plot
cmap = cm.tab20(np.linspace(0, 1, 12))
hierarchy.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])

# Fix the random seed for reproducibility
np.random.seed(0)


# ============ RC model configuration and hyperparameter values ============
config = {}

# Reservoir
config['n_internal_units'] = 450        # size of the reservoir
config['spectral_radius'] = 0.59        # largest eigenvalue of the reservoir
config['leak'] = 0.6                    # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
config['connectivity'] = 0.25           # percentage of nonzero connections in the reservoir
config['input_scaling'] = 0.1           # scaling of the input weights
config['noise_level'] = 0.01            # noise in the reservoir state update
config['n_drop'] = 5                    # transient states to be dropped
config['bidir'] = True                  # if True, use bidirectional reservoir
config['circ'] = False                  # use reservoir with circle topology

# Dimensionality reduction
config['dimred_method'] ='tenpca'       # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
config['n_dim'] = 75                    # number of resulting dimensions after the dimensionality reduction procedure

# MTS representation
config['mts_rep'] = 'reservoir'         # MTS representation:  {'last', 'mean', 'output', 'reservoir'}
config['w_ridge_embedding'] = 10.0      # regularization parameter of the ridge regression

# Readout
config['readout_type'] = None           # by setting None, the input representations will be stored

print(config)

# ============ Load dataset ============
dataset_name = 'JpVow'
data = scipy.io.loadmat('../dataset/'+dataset_name+'.mat')
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

print('Loaded '+dataset_name+' - data shape: '+ str(X.shape))

# ============ Initialize and fit the RC model ============
rcm =  RC_model(
                reservoir=None,     
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
                readout_type=config['readout_type'] 
                )

# Generate representations of the input MTS
training_time = rcm.train(X)
mts_representations = rcm.input_repr
print("Training time: %.2f seconds"%training_time)

# Compute a similarity matrix from the cosine similarity of the representations
similarity_matrix = cosine_similarity(mts_representations)
        
# Normalize the similarity in [0,1]
similarity_matrix = (similarity_matrix + 1.0)/2.0

# Plot similarity matrix
fig =  plt.figure(figsize=(5,5))
h = plt.imshow(similarity_matrix)
plt.title("RC similarity matrix")
plt.colorbar(h)
plt.show()

# Dimensionality reduction with Kernel PCA
kpca = KernelPCA(n_components=2, kernel='precomputed')
embeddings_pca = kpca.fit_transform(similarity_matrix)
plt.scatter(embeddings_pca[:,0], embeddings_pca[:,1], c=Y[:,0], s=3)
plt.title("PCA embeddings")
plt.show()

# Compute Dissimilarity matrix
Dist = 1.0 - similarity_matrix
np.fill_diagonal(Dist, 0) # due to numerical errors, the diagonal might not be 0

# Hierarchical clustering
distArray = ssd.squareform(Dist)
Z = linkage(distArray, 'ward')
clust = fcluster(Z, t=2.0, criterion="distance")
print("Found %d clusters"%len(np.unique(clust)))

# Evaluate the agreement between class and cluster labels
nmi = v_measure_score(Y[:,0], clust)
print("Normalized Mutual Information (v-score): %.3f"%nmi)

# Plot dendrogram
fig = plt.figure(figsize=(20, 10))
dn = dendrogram(Z, color_threshold=2.0, labels=None, above_threshold_color='k')
plt.show()
print("N. clusters: ", np.unique(dn['color_list']).shape[0]-1)