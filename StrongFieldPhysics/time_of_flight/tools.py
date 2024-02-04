import numpy as np


def construct_time_axis(dt:float, N:int):
    # Warning: The constructed time array is different than the one in the data file. Thus different T0!
    """Constructs time axis from time step and number of bins
    Args:
        dt (float): time step (TDC resolution. eg. 0.151 ns)
        N (int): number of bins
    """
    return np.arange(N) * dt



def select_energy_range(E:np.ndarray, Ie:np.ndarray, E_min:float, E_max:float):
    """Selects energy range from E_min to E_max
    Args:
        E (np.ndarray): Energy array
        Ie (np.ndarray): Intensity array
        E_min (float): Minimum energy
        E_max (float): Maximum energy
    Returns:
        E (np.ndarray): Energy array
        Ie (np.ndarray): Intensity array
    """
    ind = (E > E_min) & (E < E_max)
    return E[ind], Ie[ind]