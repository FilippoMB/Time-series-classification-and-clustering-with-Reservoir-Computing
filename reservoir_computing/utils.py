import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

def compute_test_scores(pred_class, Yte):
    """
    Wrapper to compute classification accuracy and F1 score

    Parameters
    ----------
    pred_class : array
        Predicted class labels
    Yte : array
        True class labels

    Returns
    -------
    accuracy : float
        Classification accuracy
    f1 : float
        F1 score
    """
    
    true_class = np.argmax(Yte, axis=1)
    
    accuracy = accuracy_score(true_class, pred_class)
    if Yte.shape[1] > 2:
        f1 = f1_score(true_class, pred_class, average='weighted')
    else:
        f1 = f1_score(true_class, pred_class, average='binary')

    return accuracy, f1


def forecasting_datasets(X, 
                         horizon, 
                         test_percent = 0.15, 
                         val_percent = 0.0, 
                         scaler = None):
    """
    This function does the following:
    1. Splits the dataset in training, validation and test sets
    2. Shift the target data by 'horizon' to create the forecasting problem
    3. Normalizes the data

    Parameters
    ----------
    X : array
        Input data
    horizon : int
        Forecasting horizon
    test_percent : float
        Percentage of the data to be used for testing
    val_percent : float
        Percentage of the data to be used for validation
        If 0, no validation set is created
    scaler : a scaler object from sklearn.preprocessing 
        Scaler object to normalize the data
        If None, a StandardScaler is created

    Returns
    -------
    Xtr : array
        Training input data
    Ytr : array
        Training target data
    Xte : array 
        Test input data
    Yte : array
        Test target data
    scaler : a scaler object from sklearn.preprocessing
        Scaler object used to normalize the data
    Xval : array (optional)
        Validation input data
    Yval : array (optional)
        Validation target data
    """
    n_data, _ = X.shape

    n_te = np.ceil(test_percent*n_data).astype(int)
    n_val = np.ceil(val_percent*n_data).astype(int)
    n_tr = n_data - n_te - n_val

    # Split dataset in training, validation and test
    tr = X[:n_tr, :]
    te = X[-n_te:, :]
    if n_val > 0:
        val = X[n_tr:-n_te, :]

    # Shift target data to create forecasting problem
    Xtr = tr[:-horizon,:]
    Ytr = tr[horizon:,:]
    Xte = te[:-horizon,:]
    Yte = te[horizon:,:]
    if n_val > 0:
        Xval = val[:-horizon,:]
        Yval = val[horizon:,:]

    # Define scaler if not provided
    if scaler is None:
        scaler = StandardScaler()

    # Fit scaler on training set
    Xtr = scaler.fit_transform(Xtr)

    # Transform the rest
    Ytr = scaler.transform(Ytr)
    Xte = scaler.transform(Xte)
    if n_val > 0:
        Xval = scaler.transform(Xval)
    
    # Add constant input
    Xtr = np.concatenate((Xtr,np.ones((Xtr.shape[0],1))),axis=1)
    Xte = np.concatenate((Xte,np.ones((Xte.shape[0],1))),axis=1)
    if n_val > 0:
        Xval = np.concatenate((Xval,np.ones((Xval.shape[0],1))),axis=1)

    if n_val > 0:
        return Xtr, Ytr, Xte, Yte, Xval, Yval, scaler
    else:
        return Xtr, Ytr, Xte, Yte, scaler