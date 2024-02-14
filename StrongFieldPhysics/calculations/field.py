# calculations for the elecatric field and vector potential

import numpy as np
from StrongFieldPhysics.calculations.constants import C, H_BAR, MASS, CHARGE, EPS0


# Electric field
# CW laser Electric field
def ElectricFieldCW(t, E0, omega, phase):
    return E0 * np.sin(omega * t + phase)

# Vector potential
def VectorPotentialCW(t, A0, omega, phase):
    return A0 * np.cos(omega * t + phase)

# Electric Field of A pulse
def ElectricFieldAPulse(t, E0, omega, phase, n, T): # not tested yet
    return E0 * np.sin(omega * t + phase) * np.sin(np.pi * t / T)**n


# Intensity
def peak_intensity(E0):
    return 0.5 * EPS0 * C * E0**2

def peak_electric_field(I):#
    # I in TW/cm2
    # convert I to W/m2
    I = I * 10**12 * 10**4
    return np.sqrt(2 * I / (EPS0 * C)) # V/m