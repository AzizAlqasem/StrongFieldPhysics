import numpy as np
from StrongFieldPhysics.calculations.constants import CHARGE, C, MASS, EPS0
import warnings

def ponderomotive_energy(intensity:float, wavelength:float):
    """Calculates the ponderomotive energy of a laser pulse

    Args:
        intensity (float): laser intensity in TW/cm^2
        wavelength (float): laser wavelength in microns
    """
    if wavelength > 100:
        print("Warning: wavelength must be in microns, not nanometers")
        print("wavelength = ", wavelength)
    intensity = intensity / 100
    const = CHARGE/ (8 * np.pi**2 * EPS0 * C**3 * MASS) * 10**6  #~ 9.33
    return const * intensity * wavelength**2


def intensity_from_Up(up, wavelength):
    """Calculates the laser intensity from the ponderomotive energy
    Args:
        up (float): ponderomotive energy in eV
        wavelength (float): laser wavelength in microns
    Returns:
        float: laser intensity in TW/cm^2
    """
    const = CHARGE/ (8 * np.pi**2 * EPS0 * C**3 * MASS) * 10**6  #~ 9.33
    return up / (const * wavelength**2) * 100 # in TW/cm^2

def gamma(up, ip):
    return np.sqrt(ip/(2*up))


def photon_energy(wavelength_nm):
    """Calculates the photon energy of a laser pulse

    Args:
        wavelength_nm (float): laser wavelength in nm
    """
    return 1239.84193 / wavelength_nm

def channel_closure(up, ip, wavelength_nm):
    """n = (Up + Ip)/PhotonEnergy"""
    return (up + ip) / photon_energy(wavelength_nm)

def intensity_from_channel_closure(channel_closure, ip, wavelength_nm):
    """Calculates the laser intensity from the channel closure
    Args:
        channel_closure (float): channel closure
        ip (float): ionization potential in eV
        wavelength_nm (float): laser wavelength in nm
    Returns:
        float: laser intensity in TW/cm^2
    """
    Up = channel_closure * photon_energy(wavelength_nm) - ip
    return intensity_from_Up(Up, wavelength_nm/1000)

def I_over_the_barrier_ionization(Ip, z=1):
    """Return Intensity in TW/cm^2
    Ip: ionization potential in eV
    """
    return 4E9 * (Ip**4 / z**2) / 1e12


##### Deprecated #####
def up(intensity:float, wavelength:float):#! Deprecated, please use ponderomotive_energy
    """Calculates the ponderomotive energy of a laser pulse

    Args:
        intensity (float): laser intensity in TW/cm^2
        wavelength (float): laser wavelength in microns
    """
    warnings.warn("up is deprecated, please use ponderomotive_energy", DeprecationWarning)
    return ponderomotive_energy(intensity, wavelength)