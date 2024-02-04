# for electrons only
# Atomic units assumes: m = e = hbar = 1
from StrongFieldPhysics.calculations.constants import *
import numpy as np

# Momentum conversion constant
P_AU = ATOMIC_MASS * ATOMIC_LENGTH / ATOMIC_TIME
AU2EV = P_AU**2 / (MASS * CHARGE) # =27.2... convert Energy from atomic units to eV

# Quantity in atomic units
C_AU = C / (ATOMIC_LENGTH / ATOMIC_TIME) # Speed of light in atomic unit
LENGTH_NM_AU = 10**-9 / ATOMIC_LENGTH # atomic unit of 1 nm in length

# Time
TIME_FS_AU = ATOMIC_TIME / 10**-15 # atomic unit of 1 fs in time


def momentum_si_to_au(p:float):
    """Converts momentum from SI to atomic units
    Args:
        p (float): momentum in SI units (kg m/s)
    Returns:
        float: momentum in atomic units
    """
    return p / P_AU

def momentum_au_to_si(p:float):
    """Converts momentum from atomic units to SI
    Args:
        p (float): momentum in atomic units
    Returns:
        float: momentum in SI units (kg m/s)
    """
    return p * P_AU

def energy_au_to_ev(E:float):
    return E * AU2EV

def energy_ev_to_au(E:float):
    return E / AU2EV


def angluar_freq_au(wavelength_nm):
    """Converts wavelength in nm to angular frequency in atomic units
    Note that E = hbar * w = (1) * w. So E=w in atomic units
    """
    const = 2 * np.pi * C_AU / LENGTH_NM_AU # 45.5633... au
    return const / wavelength_nm

def time_fs_to_au(t_fs):
    # 1fs =  41.3413... au
    return t_fs / TIME_FS_AU

def debroglie_wl_angstrom_to_energy_ev(wl_angstrom):
    """Converts de Broglie wavelength in angstrom to energy in eV
    E [ev] = h^2 / (2 m e) * 10^20 / lambda_a^2
    """
    const = BLANK_CONSTANT**2 / (2 * MASS * CHARGE) * 10**20
    return const / wl_angstrom**2

def debroglie_wl_angstrom_to_momentum_au(wl_angstrom):
    """Converts de Broglie wavelength in angstrom to momentum in atomic units
    p [au] = 2 * PI * Atomic_length * 10^10 / lambda_a
    The 2 * PI is removed because the convention is to use 2pi/lambda instead of 1/lambda
    """
    # const = 2 * PI * ATOMIC_LENGTH * 10**10
    const = ATOMIC_LENGTH * 10**10
    return const / wl_angstrom

def momentum_au_to_debroglie_wl_angstrom(p_au):
    """Converts momentum in atomic units to de Broglie wavelength in angstrom
    p [au] = 2 * PI * Atomic_length * 10^10 / lambda_a
    The 2 * PI is removed because the convention is to use 2pi/lambda instead of 1/lambda
    """
    # const = 2 * PI * ATOMIC_LENGTH * 10**10
    const = ATOMIC_LENGTH * 10**10
    return const / p_au

def momentum_au_to_angstrom_inv(p_au):
    """Converts momentum in atomic units to angstrom^-1
    which is another unit of Momentum
    p [au] = 2 * PI * Atomic_length * 10^10 / lambda_a
    """
    # const = 2 * PI * ATOMIC_LENGTH * 10**10
    const = ATOMIC_LENGTH * 10**10
    return p_au / const

def momentum_angstrom_inv_to_au(p_angstrom_inv):
    """Converts momentum in angstrom^-1 to atomic units
    which is another unit of Momentum
    p [au] = 2 * PI * Atomic_length * 10^10 / lambda_a
    """
    # const = 2 * PI * ATOMIC_LENGTH * 10**10
    const = ATOMIC_LENGTH * 10**10
    return p_angstrom_inv * const