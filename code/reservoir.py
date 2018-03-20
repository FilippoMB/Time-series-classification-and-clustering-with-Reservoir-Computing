import numpy as np
from scipy import sparse
from scipy.sparse import linalg as slinalg

class Reservoir(object):
    """
    Build a reservoir and evaluate internal states
    
    Parameters:
        n_internal_units = processing units in the reservoir
        spectral_radius = largest eigenvalue of the reservoir matrix of connection weights
        leak = amount of leakage in the reservoir state update (optional)
        connectivity = percentage of nonzero connection weights
        input_scaling = scaling of the input connection weights
        noise_level = deviation of the Gaussian noise injected in the state update
    """
    
    def __init__(self, n_internal_units=100, spectral_radius=0.99, leak=None,
                 connectivity=0.3, input_scaling=0.2, noise_level=0.01):
        
        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._spectral_radius = spectral_radius
        self._leak = leak
        self._connectivity = connectivity
        self._input_scaling = input_scaling
        self._noise_level = noise_level

        # Input weights depend on input size: they are set when data is provided
        self._input_weights = None

        # Generate internal weights
        self._internal_weights = self._initialize_internal_weights(
            n_internal_units,
            connectivity,
            spectral_radius)

    def _initialize_internal_weights(self, n_internal_units,
                                     connectivity, spectral_radius):
        # The eigs function might not converge. Attempt until it does.
        convergence = False
        while (not convergence):
            # Generate sparse, uniformly distributed weights.
            internal_weights = sparse.rand(n_internal_units,
                                           n_internal_units,
                                           density=connectivity).todense()

            # Ensure that the nonzero values are
            # uniformly distributed in [-0.5, 0.5]
            internal_weights[np.where(internal_weights > 0)] -= 0.5

            try:
                # Get the largest eigenvalue
                w, _ = slinalg.eigs(internal_weights, k=1, which='LM')

                convergence = True

            except:
                continue

        # Adjust the spectral radius.
        internal_weights /= np.abs(w)/spectral_radius

        return internal_weights

    def get_states(self, X, n_drop=0, bidir=True):
        N, T, V = X.shape
        if self._input_weights is None:
            self._input_weights = 2.0*np.random.rand(self._n_internal_units, V) - 1.0

        # compute sequence of reservoir states
        states = self._compute_state_matrix(X, n_drop)
    
        # reservoir states on time reversed input
        if bidir is True:
            X_r = X[:, ::-1, :]
            states_r = self._compute_state_matrix(X_r, n_drop)
            states = np.concatenate((states, states_r), axis=2)

        return states

    def _compute_state_matrix(self, X, n_drop=0):
        N, T, _ = X.shape
        previous_state = np.zeros((N, self._n_internal_units), dtype=float)

        # Storage
        state_matrix = np.empty((N, T - n_drop, self._n_internal_units),
                                dtype=float)

        for t in range(T):
            current_input = X[:, t, :]*self._input_scaling

            # Calculate state. Add noise and apply nonlinearity.
            state_before_tanh = self._internal_weights.dot(previous_state.T) + self._input_weights.dot(current_input.T)
            
            # Add noise
            state_before_tanh += np.random.rand(self._n_internal_units, N)*self._noise_level

            # Apply nonlinearity and leakage (optional)
            if self._leak is None:
                previous_state = np.tanh(state_before_tanh).T
            else:
                previous_state = (1.0 - self._leak)*previous_state + self._leak*np.tanh(state_before_tanh).T

            # Store everything after the dropout period
            if (t > n_drop - 1):
                state_matrix[:, t - n_drop, :] = previous_state

        return state_matrix

