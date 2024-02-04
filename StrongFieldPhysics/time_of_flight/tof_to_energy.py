import numpy as np
# import numba as nb
from scipy.interpolate import interp1d


from StrongFieldPhysics.time_of_flight.tools import construct_time_axis


#expensive function (apply numba)
# @nb.njit(fastmath=False, cache=True, parallel=False) # not faster!
def t2E(time:np.ndarray, count:np.ndarray, L=0.53, t0=-1.92e-8):
    """
    converting time to energy
    time (arr) is taken from the original data (not modified)
    L is the TOF length in meters
    t0:extra time (s) that the signel spent on the electronics. TOF = Tot_t - t0.
    """
    MASS = 9.1093837015 * 10**-31  # electon mass
    CHARGE = 1.602176634 * 10**-19  # electron charge
    # get actual time of flight (remove t0 delays)
    T = time - t0

    # remove negative and 0 time
    ind = T > 0
    T = T[ind]
    I = count[ind]  # keep only bins with positive times

    # time to energy conversion
    V = L / T  # velocity in m/s
    E = MASS * V**2 / 2.  # energy bins in Joules
    E = E / CHARGE  # energy bins in eV

    # multiply by jacobian for conversion (I(E) = I(t)*E^(-3/2))
    #constants were thrown away since we only care about relative yields
    Ie = I / E**1.5 #(3. / 2.)

    # throw away high energy data just to reduce file size, also flip arrays to make low energy at index 0
    # ind = np.argmax(E < E_max)
    E = np.flip(E)
    Ie = np.flip(Ie)
    # Normalize Ie (Normalization should be separate from t2e conversion!)
    # Ie = Ie / Ie.max()
    return E, Ie

def t2E_fixed_bins(time:np.ndarray, count:np.ndarray, dE_bin:int, E_max:int, L=0.53, t0=-1.92e-8, \
                    spline_kind='linear', **kwargs):
    """
    converting time to energy
    Covenrt time of light to energy using fixed energy bins and integrating the time of flight counts
    time (arr) is taken from the original data (not modified)
    L is the TOF length in meters
    t0:extra time (s) that the signel spent on the electronics. TOF = Tot_t - t0.

    """
    MASS = 9.1093837015 * 10**-31  # electon mass
    CHARGE = 1.602176634 * 10**-19  # electron charge
    # get actual time of flight (remove t0 delays)
    T = time*1e9 - t0*1e9 # convert to ns

    # # remove negative and 0 time
    # ind = T > 0
    # T = T[ind]
    # I = count[ind]  # keep only bins with positive times
    t_max = time[-1] - t0 # sec
    E_min = 0.5 * MASS * (L/(t_max))**2 / CHARGE
    E = np.arange(E_min+dE_bin, E_max+dE_bin, dE_bin, dtype=np.float64)
    Ie = np.zeros_like(E, dtype=np.float64)
    count_sl = interp1d(T, count, kind=spline_kind)
    for i, E_i in enumerate(E):
        E_min = E_i - dE_bin/2
        E_max = E_i + dE_bin/2
        t_max = L / np.sqrt(2*E_min*CHARGE/MASS)*1e9
        t_min = L / np.sqrt(2*E_max*CHARGE/MASS)*1e9
        Ie[i] = (count_sl(t_min)+count_sl(t_max))/2 * (t_max - t_min)
    return E, Ie

def time_to_energy_axis(time:np.ndarray, L=0.53, t0=-1.92e-8):
    """Converts time to energy axis
    Args:
        time (np.ndarray): time axis
        L (float, optional): TOF length in meters.
        t0 (_type_, optional): Time zero offset.
    """
    MASS = 9.1093837015 * 10**-31  # electon mass
    CHARGE = 1.602176634 * 10**-19  # electron charge
    # get actual time of flight (remove t0 delays)
    T = time - t0
    # remove negative and 0 time
    ind = T > 0 # if the t0 is negative, this will do nothing ...
    T = T[ind]
    # time to energy conversion
    V = L / T  # velocity in m/s
    E = MASS * V**2 / 2.  # energy bins in Joules
    E = E / CHARGE  # energy bins in eV
    return E[::-1]

def counts_of_time_to_energy(counts:np.ndarray, energy:np.ndarray):
    """apply the jacobian to yield(TOF)  to convert it to yield(E).
    Energy axis is already calculated and calibrated
    Args:
        counts (np.ndarray): TOF counts
        Energy (np.ndarray): Energy axis (axis is already flipped)
    """
    res = counts[::-1].T / energy**1.5
    return res.T


############################################# Drafts #############################################
## IMPORTANT: The functions below assumes absolute T0 (not relative to the TDC calibration), therefore they should
# not be used unless a proper T0 handling is implemented. It is just unsafe to use them now and could introduce confusion!
# For example: time in the data taken by TDC2228A starts at 8.6ns. While time assumed below starts at 0ns. Both have different
# T0s.


# def count_to_energy(counts:np.ndarray, dt:float, L=0.53, t0=-1.92e-8): # no need to input time axis
#     """Converts TOF counts to energy. The time of flight is calculated from the bin number and the time step dt.
#     Args"
#         counts (np.ndarray): TOF counts
#         dt (float): time step (TDC resolution. eg. 0.151 ns)
#         L (float, optional): TOF length in meters.
#         t0 (_type_, optional): Time zero offset.
#     """
#     time = construct_time_axis(dt, len(counts))
#     return t2E(time, counts, L=L, t0=t0)


# def construct_energy_axis(dt:float, N:int, L=0.53, t0=-1.92e-8):
#     """Constructs energy axis from time step and number of bins
#     Args:
#         N (int): number of bins
#         dt (float): time step (TDC resolution. eg. 0.151 ns)
#         L (float, optional): TOF length in meters.
#         t0 (_type_, optional): Time zero offset.
#     """
#     time = construct_time_axis(dt, N)
#     return time_to_energy_axis(time, L=L, t0=t0)


# def test_all():
#     ta = np.arange(1, 100)*1e-9
#     ca = np.arange(1, 100) # random counts
#     # Correct and well tested version
#     e0, i0 = t2E(ta, ca)

#     # Test other functions
#     e1 = time_to_energy_axis(ta)
#     e2 = construct_energy_axis(1e-9, 100-1)
#     assert np.allclose(e0, e1)
#     assert np.allclose(e0, e2)
#     i1 = counts_of_time_to_energy(ca, e0)
#     assert np.allclose(i0, i1)
#     _e, i2 = count_to_energy(ca, 1e-9)
#     assert np.allclose(i0, i2)

#     # Print results
#     print("If no error was raised, all tests passed successfully")



######### Old code ####
# def t2E_fixed_bins(time:np.ndarray, count:np.ndarray, dE_bin:int, E_max:int, L=0.53, t0=-1.92e-8, \
#                    time_resolution_interp=0.01, spline_kind='linear'):
#     """
# NOT Good!
#     converting time to energy
#     Covenrt time of light to energy using fixed energy bins and summing the time of flight counts
#     time (arr) is taken from the original data (not modified)
#     L is the TOF length in meters
#     t0:extra time (s) that the signel spent on the electronics. TOF = Tot_t - t0.

#     """
#     MASS = 9.1093837015 * 10**-31  # electon mass
#     CHARGE = 1.602176634 * 10**-19  # electron charge
#     # get actual time of flight (remove t0 delays)
#     T = time*1e9 - t0*1e9 # convert to ns

#     # # remove negative and 0 time
#     # ind = T > 0
#     # T = T[ind]
#     # I = count[ind]  # keep only bins with positive times

#     E = np.arange(dE_bin, E_max, dE_bin, dtype=np.float64)
#     Ie = np.zeros_like(E, dtype=np.float64)
#     count_sl = interp1d(T, count, kind=spline_kind)
#     t_expanded = np.arange(T.min(), T.max(), time_resolution_interp, dtype=np.float64)
#     count_expanded = count_sl(t_expanded)
#     for i, E_i in enumerate(E):
#         E_min = E_i - dE_bin/2
#         E_max = E_i + dE_bin/2
#         t_max = L / np.sqrt(2*E_min*CHARGE/MASS)*1e9
#         t_min = L / np.sqrt(2*E_max*CHARGE/MASS)*1e9
#         ind = (t_expanded > t_min) & (t_expanded < t_max)
#         Ie[i] = count_expanded[ind].sum()
#     return E, Ie
