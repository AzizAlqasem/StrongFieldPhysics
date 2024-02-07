import numpy as np
from scipy.optimize import curve_fit


# Half wave plate (HWP)- Polarizer relative angle to power
def cos2_func(ang, A, Q, B):
    """Cosine squared function
    Args:
        ang (np.ndarray): angle in degrees
        A (float): amplitude
        Q (float): phase
        B (float): offset
    """
    return A * np.cos(np.deg2rad(ang)-Q)**2 + B

def cos2_func2(ang, A, Q, B, f): # better fit
    """Cosine squared function
    Args:
        ang (np.ndarray): angle in degrees
        A (float): amplitude
        Q (float): phase
        B (float): offset
        f (float): frequency
    """
    return A * np.cos(f*np.deg2rad(ang)-Q)**2 + B

def ang_to_power(ang, power, p0=[1,0,0], fit_func=cos2_func):
    """Converts angle to power
    Args:
        ang (np.ndarray): angle in degrees
        power (np.ndarray): power in mW
        p0 (list): initial guess for the fit. Defaults to [1,0,0].
        fit_func (function): fit function
    return:
        function (angles : np.ndarray): -->  power
    """
    popt, pcov = curve_fit(fit_func, ang, power, p0=p0)
    #TODO: Assert that the fit is good
    print('fit parameters: ', popt, '\n', 'covariance: ', pcov)
    return lambda angle: fit_func(angle, *popt)

ang_to_intensity = ang_to_power # alias