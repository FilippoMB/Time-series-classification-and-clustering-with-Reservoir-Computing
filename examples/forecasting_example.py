import numpy as np
from sklearn.metrics import mean_squared_error
from reservoir_computing.modules import RC_forecaster
from reservoir_computing.utils import make_forecasting_dataset
from reservoir_computing.datasets import PredLoader

np.random.seed(0) # For reproducibility

# Load the dataset
ts_full = PredLoader().get_data('ElecRome')

# Resample the time series to hourly frequency
ts_hourly = np.mean(ts_full.reshape(-1, 6), axis=1)

# Use only the first 3000 time steps
ts_small = ts_hourly[0:3000, None]

# Generate training and testing datasets
Xtr, Ytr, Xte, Yte, scaler = make_forecasting_dataset(
    ts_small, 
    horizon=24, # forecast horizon of 24h ahead
    test_percent = 0.1)

# Initialize the RC model 
forecaster =  RC_forecaster(n_internal_units=900)

# Train the model
forecaster.fit(Xtr, Ytr)

# Compute predictions on test data
Yhat = forecaster.predict(Xte)
Yhat = scaler.inverse_transform(Yhat) # Revert the scaling of the predictions 
mse = mean_squared_error(Yte, Yhat)
print(f"Mean Squared Error: {mse:.2f}")