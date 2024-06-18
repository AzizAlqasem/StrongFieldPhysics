import numpy as np


def calculate_sigma(xd, yd, fitted_function, popt):
    """Calculates the sigma of the data points
    !! WARNING: This is not the standard deviation of repeated measurements !!
    Do NOT use this with fitting error.
    Args:
        xd (np.ndarray): x data points
        yd (np.ndarray): y data points
        fitted_function (function): fitted function
        popt (np.ndarray): fitted parameters
    Returns:
        np.ndarray: sigma of the data points
    """
    return np.sqrt(np.sum((yd - fitted_function(xd, *popt))**2) / len(yd))