import requests
import numpy as np
from io import BytesIO
import warnings
import collections
from scipy.integrate import solve_ivp

class ClfLoader:
    r"""
    Class to download and load time series classification datasets.
    """
    def __init__(self) -> None:
        self.datasets = {
            'AtrialFibrillation': ('https://zenodo.org/records/10852712/files/AF.npz?download=1', 'Multivariate time series classification.\nSamples: 5008 (4823 training, 185 test)\nFeatures: 2\nClasses: 3\nTime series length: 45'),
            'ArabicDigits': ('https://zenodo.org/records/10852747/files/ARAB.npz?download=1', 'Multivariate time series classification.\nSamples: 8800 (6600 training, 2200 test)\nFeatures: 13\nClasses: 10\nTime series length: 93'),
            'Auslan': ('https://zenodo.org/records/10839959/files/Auslan.npz?download=1', 'Multivariate time series classification.\nSamples: 2565 (1140 training, 1425 test)\nFeatures: 22\nClasses: 95\nTime series length: 136'),
            'CharacterTrajectories': ('https://zenodo.org/records/10852786/files/CHAR.npz?download=1', 'Multivariate time series classification.\nSamples: 2858 (300 training, 2558 test)\nFeatures: 3\nClasses: 20\nTime series length: 205'),
            'CMUsubject16': ('https://zenodo.org/records/10852831/files/CMU.npz?download=1', 'Multivariate time series classification.\nSamples: 58 (29 training, 29 test)\nFeatures: 62\nClasses: 2\nTime series length: 580'),
            'ECG2D': ('https://zenodo.org/records/10839881/files/ECG_2D.npz?download=1', 'Multivariate time series classification.\nSamples: 200 (100 training, 100 test)\nFeatures: 2\nClasses: 2\nTime series length: 152'),
            'Japanese_Vowels': ('https://zenodo.org/records/10837602/files/Japanese_Vowels.npz?download=1', 'Multivariate time series classification.\nSamples: 640 (270 training, 370 test)\nFeatures: 12\nClasses: 9\nTime series length: 29'),
            'KickvsPunch': ('https://zenodo.org/records/10852865/files/KickvsPunch.npz?download=1', 'Multivariate time series classification.\nSamples: 26 (16 training, 10 test)\nFeatures: 62\nClasses: 2\nTime series length: 841'),
            'Libras': ('https://zenodo.org/records/10852531/files/LIB.npz?download=1', 'Multivariate time series classification.\nSamples: 360 (180 training, 180 test)\nFeatures: 2\nClasses: 15\nTime series length: 45'),
            'NetFlow': ('https://zenodo.org/records/10840246/files/NET.npz?download=1', 'Multivariate time series classification.\nSamples: 1337 (803 training, 534 test)\nFeatures: 4\nClasses: 2\nTime series length: 997'),
            'RobotArm': ('https://zenodo.org/records/10852893/files/Robot.npz?download=1', 'Multivariate time series classification.\nSamples: 164 (100 training, 64 test)\nFeatures: 6\nClasses: 5\nTime series length: 15'),
            'UWAVE': ('https://zenodo.org/records/10852667/files/UWAVE.npz?download=1', 'Multivariate time series classification.\nSamples: 628 (200 training, 428 test)\nFeatures: 3\nClasses: 8\nTime series length: 315'),
            'Wafer': ('https://zenodo.org/records/10839966/files/Wafer.npz?download=1', 'Multivariate time series classification.\nSamples: 1194 (298 training, 896 test)\nFeatures: 6\nClasses: 2\nTime series length: 198'),
            'Chlorine': ('https://zenodo.org/records/10840284/files/CHLO.npz?download=1', 'Univariate time series classification.\nSamples: 4307 (467 training, 3840 test)\nFeatures: 1\nClasses: 3\nTime series length: 166'), 
            'Phalanx': ('https://zenodo.org/records/10852613/files/PHAL.npz?download=1', 'Univariate time series classification.\nSamples: 539 (400 training, 139 test)\nFeatures: 1\nClasses: 3\nTime series length: 80'),
            'SwedishLeaf': ('https://zenodo.org/records/10840000/files/SwedishLeaf.npz?download=1', 'Univariate time series classification.\nSamples: 1125 (500 training, 625 test)\nFeatures: 1\nClasses: 15\nTime series length: 128'),
        }

    def available_datasets(self, details=False):
        r"""
        Print the available datasets.

        Parameters:
        -----------
        details : bool
            If True, print a description of the datasets.

        Returns:
        --------
        None
        """
        print("Available datasets:\n")
        for alias, (_, description) in self.datasets.items():
            if details:
                print(f"{alias}\n-----------\n{description}\n")
            else:
                print(alias)

    def get_data(self, alias):
        r"""
        Download and load the dataset.

        Parameters:
        -----------
        alias : str
            The alias of the dataset to be downloaded.

        Returns:
        --------
        Xtr : np.ndarray
            Training data
        Ytr : np.ndarray
            Training labels
        Xte : np.ndarray
            Test data
        Yte : np.ndarray
            Test labels
        """

        if alias not in self.datasets:
            raise ValueError(f"Dataset {alias} not found.")

        url, _ = self.datasets[alias]
        response = requests.get(url)
        if response.status_code == 200:

            data = np.load(BytesIO(response.content))
            Xtr = data['Xtr']  # shape is [N,T,V]
            if len(Xtr.shape) < 3:
                Xtr = np.atleast_3d(Xtr)
            Ytr = data['Ytr']  # shape is [N,1]
            Xte = data['Xte']
            if len(Xte.shape) < 3:
                Xte = np.atleast_3d(Xte)
            Yte = data['Yte']
            n_classes_tr = len(np.unique(Ytr))
            n_classes_te = len(np.unique(Yte))
            if n_classes_tr != n_classes_te:
                warnings.warn(f"Number of classes in training and test sets do not match for {alias} dataset.")
            print(f"Loaded {alias} dataset.\nNumber of classes: {n_classes_tr}\nData shapes:\n  Xtr: {Xtr.shape}\n  Ytr: {Ytr.shape}\n  Xte: {Xte.shape}\n  Yte: {Yte.shape}")

            return (Xtr, Ytr, Xte, Yte)
        else:
            print(f"Failed to download {alias} dataset.")
            return None


class PredLoader():
    """
    Class to download and load time series forecasting datasets.
    """
    def __init__(self) -> None:
        self.datasets = {
            'ElecRome': ('https://zenodo.org/records/10910985/files/Elec_Rome.npz?download=1', 'Univariate time series forecasting.\nLength: 137376\nFeatures: 1'),
            'CDR': ('https://zenodo.org/records/10911142/files/CDR.npz?download=1', 'Multivariate time series forecasting.\nLength: 3336\nFeatures: 8'),
        }

    def available_datasets(self, details=False):
        """
        Print the available datasets.

        Parameters:
        -----------
        details : bool
            If True, print a description of the datasets.

        Returns:
        --------
        None
        """
        print("Available datasets:\n")
        for alias, (_, description) in self.datasets.items():
            if details:
                print(f"{alias}\n-----------\n{description}\n")
            else:
                print(alias)

    def get_data(self, alias) -> np.ndarray:
        """
        Download and load the dataset.

        Parameters:
        -----------
        alias : str
            The alias of the dataset to be downloaded.

        Returns:
        --------
        X : np.ndarray
            Time series data
        """
        if alias not in self.datasets:
            raise ValueError(f"Dataset {alias} not found.")

        url, _ = self.datasets[alias]
        response = requests.get(url)
        if response.status_code == 200:

            data = np.load(BytesIO(response.content))
            X = data['X']
            print(f"Loaded {alias} dataset.\nData shape:\n  X: {X.shape}")

            return X

        else:
            print(f"Failed to download {alias} dataset.")
            return None


def mackey_glass(sample_len=1000, tau=17, delta_t=1, seed=None, n_samples = 1):
    r"""Generate the Mackey Glass time-series. 

        Parameters:
        -----------
        sample_len : int (default ``1000``)
            Length of the time-series in timesteps.
        tau : int (default ``17``)
            Delay of the MG system. Commonly used values are tau=17 (mild 
            chaos) and tau=30 (moderate chaos).
        delta_t : int (default ``1``)
            Time step of the simulation.
        seed : int or None (default ``None``)
            Seed of the random generator. Can be used to generate the same
            timeseries each time.
        n_samples : int (default ``1``)
            Number of samples to generate.

        Returns:
        --------
        np.ndarray | list
            Generated Mackey-Glass time-series.
            If n_samples is 1, a single array is returned. Otherwise, a list.
    """
    np.random.seed(seed)
    history_len = tau * delta_t 
    
    # Initial condition
    timeseries = 1.2
    
    samples = []
    for _ in range(n_samples):
        history = collections.deque(1.2 * np.ones(history_len) + 0.2 * \
                                    (np.random.rand(history_len) - 0.5))
        # Preallocate the array for the time-series
        inp = np.zeros((sample_len,1))
        
        for timestep in range(sample_len):
            for _ in range(delta_t):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) - \
                             0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries
        
        # Squash timeseries through tanh
        inp = np.tanh(inp - 1)
        samples.append(inp)

    if n_samples == 1:
        return samples[0]
    else:
        return samples


def mso(T=1000, N=10, seed=None, freq=0.5):
    r"""Generates the Multiple Sinewave Oscillator (MSO) time-series 
    by combining inusoids with incommensurable periods.
    The sinusoids to be combined are selected randomly.

    Parameters:
    -----------
    T : int (default ``1000``)
        Number of time steps.
    N : int (default ``10``)
        Maximum number of sinusoids to combine.
    seed : int or None (default ``None``)
        Seed for the random generator.
    freq : float (default ``0.5``)
        Frequency of the sinusoids.

    Returns:
    --------
    np.ndarray
        MSO time-series.
    """
    np.random.seed(seed)

    t = np.arange(T * freq, step=freq)
    print(f"MSO - signal frequencies:")
    print(f"  min period: {2 * np.pi * (1 / freq):.2f}")
    print(f"  max period: {np.exp((N - 1) / N) * 2 * np.pi * (1 / freq):.2f}")
    x_t = np.arange(N)
    base_sinusoids = np.sin(1 / np.exp(x_t / N)[:, None] @ t[None])
    
    mixer = np.random.choice([0, 1], size=(1,N), p=[0.5, 0.5])
    np.random.seed(None)
    X = mixer@base_sinusoids

    return X.T


def _lorenz_system(t, y, sigma, rho, beta):
    """Lorenz system of differential equations.
    """
    x, y, z = y
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def lorenz(sigma=10, rho=28, beta=8/3, y0=[0, -0.01, 9.0], t_span=[0, 100], dt=1e-3):
    r"""Generate the Lorenz attractor time-series.
    
    Parameters:
    -----------
    sigma : float (default ``10``)
        1st parameter of the Lorenz system.
    rho : float (default ``28``)
        2nd parameter of the Lorenz system.
    beta : float (default ``8/3``)
        3rd parameter of the Lorenz system.
    y0 : list (default ``[0, -0.01, 9.0]``)
        Initial conditions of the Lorenz system.
    t_span : list (default ``[0, 100]``)
        Time span of the simulation.
    dt : float (default ``1e-3``)
        Time step of the simulation.

    Returns:
    --------
    np.ndarray
        Lorenz time-series.
    """
    t = np.linspace(t_span[0], t_span[1], int(1/dt))
    solution = solve_ivp(_lorenz_system, t_span, y0, args=(sigma, rho, beta), t_eval=t)
    return solution.y.T


def _rossler_system(t, y, a, b, c):
    """Rossler system of differential equations.
    """
    x, y, z = y
    dxdt = -y - z
    dydt = x + a*y
    dzdt = b + z*(x - c)
    return [dxdt, dydt, dzdt]


def rossler(a=0.2, b=0.2, c=5.7, y0=[0.5, 0.5, 0.5], t_span=[0, 200], dt=1e-3):
    r"""Generate the Rossler attractor time-series.
    
    Parameters:
    -----------
    a : float (default ``0.2``)
        1st parameter of the Rossler system.
    b : float (default ``0.2``)
        2nd parameter of the Rossler system.
    c : float (default ``5.7``)
        3rd parameter of the Rossler system.
    y0 : list (default ``[0, 0.1, 0]``)
        Initial conditions of the Rossler system.
    t_span : list (default ``[0, 100]``)
        Time span of the simulation.
    dt : float (default ``1e-3``)
        Time step of the simulation.

    Returns:
    --------
    np.ndarray
        Rossler time-series.
    """
    t = np.linspace(t_span[0], t_span[1], int(1/dt))
    solution = solve_ivp(_rossler_system, t_span, y0, args=(a, b, c), t_eval=t)
    return solution.y.T


class SynthLoader:
    """
    Class to generate synthetic time series.
    """

    def __init__(self) -> None:
        self.datasets = {
            'MG': (mackey_glass, 'Generate the Mackey Glass time-series'),
            'MSO': (mso, 'Generate the Multiple Superimposed Oscillator time-series'),
            'Lorenz': (lorenz, 'Generate the Lorenz attractor time-series'),
            'Rossler': (rossler, 'Generate the Rossler attractor time-series'),
        }

    def available_datasets(self, details=False):
        """
        Print the available synthetic datasets.

        Returns:
        --------
        None
        """
        print("Available synthetic datasets:\n")
        for alias, (_, description) in self.datasets.items():
            if details:
                print(f"{alias}\n-----------\n{description}\n")
            else:
                print(alias)

    def get_data(self, alias, **kwargs):
        """
        Generate the synthetic time series.

        Parameters:
        -----------
        alias : str
            The alias of the synthetic dataset to be generated.
        kwargs : dict
            Additional parameters for the synthetic dataset.

        Returns:
        --------
        np.ndarray
            Synthetic time series.
        """
        if alias not in self.datasets:
            raise ValueError(f"Dataset {alias} not found.")

        generator, _ = self.datasets[alias]
        X = generator(**kwargs)
        print(f"Generated {alias} dataset.\nData shape:\n  X: {X.shape}")

        return X


if __name__ == '__main__':
    # Example usage (classification)
    downloader = ClfLoader()
    downloader.available_datasets(details=False)  # Print available datasets
    Xtr, Ytr, Xte, Yte = downloader.get_data('Libras')  # Download dataset and return data

    # Example usage (forecasting)
    downloader = PredLoader()
    downloader.available_datasets(details=False)  # Print available datasets
    X = downloader.get_data('CDR')  # Download dataset and return data

    # Example usage (synthetic)
    synth = SynthLoader()
    synth.available_datasets()  # Print available datasets
    Xs = synth.get_data('Lorenz')  # Generate synthetic time series