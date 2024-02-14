import numpy as np


def power_to_intensity(power, intensity, return_fit=False):
    """Converts power to intensity using linear fit)
    Args:
        power (np.ndarray): power in mW
        intensity (np.ndarray): intensity in W/cm^2
    Returns:
        function (power : np.ndarray): -->  intensity
    """
    popt, pcov = np.polyfit(power, intensity, 1, cov=True)
    if return_fit:
        print('fit parameters: ', popt, '\n', 'covariance: ', pcov)
        return lambda p: np.poly1d(popt)(p), popt, pcov
    return lambda p: np.poly1d(popt)(p)
