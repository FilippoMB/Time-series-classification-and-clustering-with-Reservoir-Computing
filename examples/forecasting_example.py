import requests
from io import BytesIO
import pprint
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error

from reservoir_computing.modules import RC_forecaster
from reservoir_computing.utils import forecasting_datasets

# Fix the random seed for reproducibility
np.random.seed(0)

# ============ RC model configuration and hyperparameter values ============
config = {}

# Hyperarameters of the reservoir
config['n_internal_units'] = 900        # size of the reservoir
config['spectral_radius'] = 0.95        # largest eigenvalue of the reservoir
config['leak'] = None                   # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
config['connectivity'] = 0.25           # percentage of nonzero connections in the reservoir
config['input_scaling'] = 0.1           # scaling of the input weights
config['noise_level'] = 0.0             # noise in the reservoir state update
config['n_drop'] = 10                   # transient states to be dropped
config['circ'] = False                  # use reservoir with circle topology

# Dimensionality reduction hyperparameters
config['dimred_method'] = None          # options: {None (no dimensionality reduction), 'pca'}
config['n_dim'] = 75                    # number of resulting dimensions after the dimensionality reduction procedure

# Linear readout hyperparameters
config['w_ridge'] = 1.0                 # regularization of the ridge regression readout

pprint.pprint(config)

# ============ Load dataset =================
data_url = 'https://raw.githubusercontent.com/FilippoMB/lecture_RNN_phase_space/master/TS_Acea.mat'
response = requests.get(data_url)
response.raise_for_status()
data = scipy.io.loadmat(BytesIO(response.content))
ts_full = data["X"]

# Resample the time series to hourly frequency
ts_hourly = np.mean(ts_full.reshape(-1, 6), axis=1)

# Use only the first 3000 time steps
ts_small = ts_hourly[0:3000, None]

# Generate training and testing datasets
Xtr, Ytr, Xte, Yte, scaler = forecasting_datasets(
    ts_small, 
    horizon=24, # forecast horizon of 24h ahead
    test_percent = 0.1)

# ============ Initialize, train and evaluate the RC model ============
forecaster =  RC_forecaster(reservoir=None,     
                            n_internal_units=config['n_internal_units'],
                            spectral_radius=config['spectral_radius'],
                            leak=config['leak'],
                            connectivity=config['connectivity'],
                            input_scaling=config['input_scaling'],
                            noise_level=config['noise_level'],
                            circle=config['circ'],
                            n_drop=config['n_drop'],
                            dimred_method=config['dimred_method'], 
                            n_dim=config['n_dim'],          
                            w_ridge=config['w_ridge'])

# Train the model
forecaster.fit(Xtr, Ytr)

# Compute predictions on test data
Yhat = forecaster.predict(Xte)
Yhat = scaler.inverse_transform(Yhat) # We need to reverse the scaling of the predictions 
mse = mean_squared_error(Yte[config['n_drop']:,:], Yhat)
print(f"Mean Squared Error: {mse:.2f}")