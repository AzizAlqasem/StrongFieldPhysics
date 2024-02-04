import numpy as np

def up(intensity:float, wavelength:float):
    """Calculates the ponderomotive energy of a laser pulse

    Args:
        intensity (float): laser intensity in TW/cm^2
        wavelength (float): laser wavelength in microns
    """
    if wavelength > 100:
        print("Warning: wavelength must be in microns, not nanometers")
        print("wavelength = ", wavelength)
    intensity = intensity / 100
    return 9.33 * intensity * wavelength**2


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