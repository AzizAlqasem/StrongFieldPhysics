# CLASSICAL OSCILLATOR APPROACH to study AC Stark shift
# paper: Five ways to the nonresonant dynamic Stark effect, 2011

import numpy as np
import matplotlib.pyplot as plt
# from StrongFieldPhysics.calculations.atomic_unit import energy_au_to_ev, angluar_freq_au, energy_ev_to_au
from StrongFieldPhysics.calculations.constants import C, H_BAR, MASS, CHARGE
from StrongFieldPhysics.calculations.field import peak_electric_field
from StrongFieldPhysics.calculations.calculations import ponderomotive_energy

# Electric field
def E(t, E0, omega, phase):
    return E0 * np.sin(omega * t + phase)

# Potential energy
# V = 1/2 m w^2 x^2 - q x E(t)
def V(t, x, m, w, q, E0, omega, phase):
    return 1/2 * m * w**2 * x**2 - q * x * E(t, E0, omega, phase)

# Position X(t)
# x(t) = x0 * cos(wt + phase)
# x0 = (qE0/m) / (w_i^2 - w^2)
def x(t, x0, omega, phase):
    return x0 * np.cos(omega * t + phase)

def x0(m, q, E0, w, wi):
    return (q * E0 / m) / (wi**2 - w**2)

# Dipole moment
# u(t) = q x(t)
def u_x(t, q, x0, omega, phase):
    return q * x(t, x0, omega, phase)

# u(t) = a E(t), where a is the polarizability
def u_E(t, a, E0, omega, phase):
    return a * E(t, E0, omega, phase)

def a(m, q, w, wi): # polarizability
    # w is the angular frequency of the electric field
    # wi is the angular frequency of the energy level
    # m is the mass of the particle and q is the charge
    return q**2 / (m * (wi**2 - w**2))

# Energy of the induced dipole = -<1/2u(t)E(t)> = -1/4 a E0^2
def induced_dipole_energy(a, E0):
    return -1/4 * a * E0**2


##### Calculations #####
### Stark shift = - 1/4 a E0^2
# input: laser intensity [TW/cm2], wavelength [um], energy level (energy in ev), ionization potential [ev]
# output: Stark shift in ev [induced_dipole_energy]
def classical_ac_stark_shift(intensity:float, wavelength:float, energy_level:float, Ip:float, \
                             dont_shift_at_resonance=False, resonance_width_pe=2, rydberg_limit_pe=0.5,
                             max_shift_is_up=True):
    # resonance_width_pe: resonance width in multiple of photon enrgy
    # rydberg_limit_pe: rydberg limit in multiple of photon energy. Any level between Ip and Ip+rydberg_limit_pe*pe will be shifted as Up
    E0 = peak_electric_field(intensity)
    wl_m = wavelength * 10**-6
    omega = 2 * np.pi * C / wl_m
    pe = H_BAR * omega / CHARGE
    ire = energy_level - Ip # ionization resonance energy
    if ire > 0:
        raise ValueError(f"Energy level {energy_level:.2f}ev must be less than the ionization potential {Ip:.2f}ev")
    if max_shift_is_up==False: # then use another method to avoid the resonance case
        if abs(ire) < pe * rydberg_limit_pe:
            ire = 0 # set the energy level to the ionization potential such that the shift is Up (ponderomotive energy)
        elif dont_shift_at_resonance and abs(ire) < resonance_width_pe*pe:
            print(f"Warning! This energy level {energy_level:.2f}ev is within the resonance width {resonance_width_pe*pe:.2f}ev, therefore shift is set to Zero.")
            return 0.0 # no shift at resonance since the classical oscillator model is not valid at resonance
        if dont_shift_at_resonance==False:
            print(f"Warning! Classical calculation of AC Stark shift should be far from resonance {pe:.2f}ev")
    # convert energy level angular frequency in SI units
    # convert energy from ev to si
    eng_j = ire * CHARGE
    level_angfreq = eng_j / H_BAR
    # calculate polarizability
    polarizability = a(MASS, CHARGE, omega, level_angfreq)
    # calculate induced dipole energy
    stark_shift = induced_dipole_energy(polarizability, E0) / CHARGE
    if max_shift_is_up:
        up = ponderomotive_energy(intensity, wavelength)
        stark_shift = np.sign(stark_shift) * min(abs(stark_shift), abs(up))
    return stark_shift

