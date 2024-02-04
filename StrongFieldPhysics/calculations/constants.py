# Atomic units assumptions: m = e = hbar = 1

import math

PI = math.pi

MASS = 9.1093837015 * 10**-31  # electon mass
CHARGE = 1.602176634 * 10**-19  # electron charge
BOHR_RADIUS = 5.29177210903e-11 # Meter  # Bohr radius
C = 299792458 # Speed of light in m/s
BLANK_CONSTANT = 6.62608e-34 # J s  # Planck constant
H_BAR = BLANK_CONSTANT / (2 * PI) # J s  # Reduced Planck constant

# One Atomic units (in SI units)
ATOMIC_MASS = MASS # atomic unit of mass
ATOMIC_LENGTH = BOHR_RADIUS # Meter  # atomic unit of length
ATOMIC_TIME = 2.418884326505e-17 # Second  # atomic unit of time
ATOMIC_CHARGE = CHARGE # atomic unit of charge

# Classical trajectories
LONG_SHORT_TRAJ_BORN_PHASE_BOUNDARY = 1.884196504 # radian