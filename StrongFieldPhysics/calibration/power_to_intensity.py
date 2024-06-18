import numpy as np
# least square fit
import scipy.optimize as opt
from StrongFieldPhysics.tools.statistics import calculate_sigma

def power_to_intensity(power, intensity, return_fit=False, return_sigma=False):
    """Converts power to intensity using linear fit)
    Args:
        power (np.ndarray): power in mW
        intensity (np.ndarray): intensity in W/cm^2
    Returns:
        function (power : np.ndarray): -->  intensity
    """
    if return_fit:
        popt, pcov = np.polyfit(power, intensity, 1, cov=True)
        print('fit parameters: ', popt, '\n', 'covariance: ', pcov)
        return lambda p: np.poly1d(popt)(p), popt, pcov
    elif return_sigma:
        popt, residual, *_ = np.polyfit(power, intensity, 1, full=True)
        sigma = np.sqrt(residual[0] / (len(intensity) - 1))
        return lambda p: np.poly1d(popt)(p), popt, sigma
    else:
        popt = np.polyfit(power, intensity, 1)
        return lambda p: np.poly1d(popt)(p)

def power_to_intensity_slop(power, intensity): # Onlt slope : I(p) = a * p
    """Converts power to intensity using linear fit with least square fit)
    Args:
        power (np.ndarray): power in mW
        intensity (np.ndarray): intensity in TW/cm^2
    Returns:
        function (power : np.ndarray): -->  intensity
    """
    func = lambda x, a: a * x
    popt, pcov = opt.curve_fit(func, power, intensity)
    sigma = np.sqrt(pcov[0][0])# does not make sense! Too small
    # sigma = calculate_sigma(power, intensity, func, popt) # Wrong! because this is only valied for repeated measurements not fitting
    return lambda p: popt[0] * p, popt, sigma