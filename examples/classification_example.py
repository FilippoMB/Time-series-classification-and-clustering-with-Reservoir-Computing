import numpy as np
from sklearn.preprocessing import OneHotEncoder
from reservoir_computing.modules import RC_model
from reservoir_computing.utils import compute_test_scores
from reservoir_computing.datasets import ClfLoader

np.random.seed(0) # Fix the seed for reproducibility

# Load the data
Xtr, Ytr, Xte, Yte = ClfLoader().get_data('Japanese_Vowels')

# One-hot encoding for labels
onehot_encoder = OneHotEncoder(sparse_output=False)
Ytr = onehot_encoder.fit_transform(Ytr)
Yte = onehot_encoder.transform(Yte)

# Initialize, train and evaluate the RC model 
classifier =  RC_model(n_internal_units=500)

# Train the model
tr_time = classifier.fit(Xtr, Ytr) 

# Compute predictions on test data
pred_class = classifier.predict(Xte) 
accuracy, f1 = compute_test_scores(pred_class, Yte)
print(f"Accuracy = {accuracy:.3f}, F1 = {f1:.3f}")