import numpy as np

from StrongFieldPhysics.calculations.constants import MASS, CHARGE
from StrongFieldPhysics.calculations.atomic_unit import momentum_si_to_au, momentum_au_to_si, energy_au_to_ev


def t2P(time:np.ndarray, counts:np.ndarray, L=0.53, t0=-1.92e-8):
    p = time_to_momentum_axis(time, L, t0)
    I_p = counts[::-1] / p**2
    return p, I_p

def p2E(p:np.ndarray, I_p:np.ndarray):
    # Assuming p is in atomic units
    # p = momentum_au_to_si(p)
    # E = p**2 / (2 * MASS) / CHARGE # E in eV
    E = p**2 / 2
    E = energy_au_to_ev(E)
    I_E = I_p / np.sqrt(p)
    return E, I_E

def time_to_momentum_axis(time:np.ndarray, L=0.53, t0=-1.92e-8):
    """Converts time to momentum axis
    Args:
        time (np.ndarray): time axis in seconds
        L (float, optional): TOF length in meters.
        t0 (_type_, optional): Time zero offset.
    """
    # get actual time of flight (remove t0 delays)
    T = time - t0
    # # remove negative and 0 time
    # ind = T > 0 # if the t0 is negative, this will do nothing ...
    # T = T[ind]
    # time to energy conversion
    V = L / T  # velocity in m/s
    P = MASS * V  # momentum bins in kg m/s
    P =  momentum_si_to_au(P)  # momentum bins in au
    return P[::-1]

def momentum_to_energy_axis(p:np.ndarray):
    # p = momentum_au_to_si(p)
    # Assuming momentum is in atomic units
    E_au = p**2/2
    return energy_au_to_ev(E_au) #p**2 / (2 * MASS) / CHARGE # E in eV

def counts_of_time_to_momentum(counts:np.ndarray, momentum:np.ndarray):
    """apply the jacobian to yield(TOF)  to convert it to yield(momentum).
    Momentum axis should be already calculated and calibrated
    Args:
        counts (np.ndarray): TOF counts
        momentum (np.ndarray): momentum axis (axis is already flipped)
    """
    res = counts[::-1].T / momentum**2
    return res.T

def momentum_au_to_tof_ns(p_au, t0_ns, L) -> float:
    """convert momentum in atomic units to time of flight in nano seconds.

    Args:
        p_au (float): momentum in atomic units
        t0_ns (float): calibrated T0 in nano seconds
        L (flaot): length of the time of flight in meters

    Returns:
        float: time of flight in nano seconds
    """
    p = momentum_au_to_si(p_au)
    v = p / MASS
    t = L / v
    # convert to nano seconds
    t = t * 10**9
    return t + t0_ns