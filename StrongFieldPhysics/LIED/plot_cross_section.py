import numpy as np
import matplotlib.pyplot as plt
import re
import os

from StrongFieldPhysics.parser.data_files import read_header
from StrongFieldPhysics.time_of_flight.tof_to_momentum import time_to_momentum_axis
from StrongFieldPhysics.LIED.cross_section import find_crossection, search_traj_table, find_crossection2, find_crossection3
from StrongFieldPhysics.calculations.constants import LONG_SHORT_TRAJ_BORN_PHASE_BOUNDARY



def plot_dcs(E, Up, delta, t0, L, E0f, WL, I0, TARGET, SAVE,\
             time_arr_path=None, has_theory=False, thory_label='CED', theory_path=None,\
             ylim=None, xlim=None, trajectroy_type='long', theory_y_offset=0):

    born_phase = search_traj_table("born_phase", "ke_scatter", E/Up, trajectroy_type=trajectroy_type)
    print(f"Born phase = {born_phase}")
    # Check if the born phase is within the long trajectories
    if born_phase > LONG_SHORT_TRAJ_BORN_PHASE_BOUNDARY: # Look at Cosmin thesis page 21 Fig.1.4
        print(f"\n!!Warning!!\nBorn phase = {born_phase} is NOT within the long trajectories\n")
    # Read 2d array from file where the columns are the angle and the rows are the time of flight bins
    path = f"{TARGET}_total_spectra_TDC2228A.dat"
    angle_header = read_header(path)[0] ## Angle (Columns): 20 to 40 step 0.2
    first_angle, last_angle, angle_step = [int(ang) for ang in re.findall(r'(-?\d+.?\d?)', angle_header)]
    total_spectra_arr = np.loadtxt(path, delimiter='\t')
    # divide each spectrum by the number of laser shots
    path = f"{TARGET}_averaged_count_TDC2228A.dat"
    angles, avg_count, tot_laser_shots = np.loadtxt(path).T
    #*WARNING*  make sure that angles matches the count, and angles are in ascending order!
    total_spectra_arr /= tot_laser_shots[None, :]


    # dt = 0.151E-9 # time bin width for TDC2228A
    if time_arr_path is None:
        time_arr_path = f"{TARGET}_{WL}um_Round1_ang00_TDC2228A.csv"
    time_arr = np.loadtxt(time_arr_path, delimiter=',', usecols=0) *1e-9 # all files have the same time axis
    p = time_to_momentum_axis(time_arr, L=L, t0=t0) # magnitude of momentum in a.u.

    # Loop over Angles and get parrallel and perpendicular momentum for each angle
    angles = np.arange(first_angle, last_angle+angle_step, angle_step)
    dcs_LIED, scattering_angles_LIED, calculations_dict = find_crossection(total_spectra_arr, p, angles, born_phase, Up, delta, trajectroy_type=trajectroy_type)
    # print(dcs_LIED)
    # normalize
    dcs_LIED /= np.max(dcs_LIED[:4])
    error = 1 / np.sqrt(dcs_LIED *tot_laser_shots[None, :])
    ### ------------------- Plotting ------------------- ###
    fig, ax = plt.subplots()
    ax.errorbar(scattering_angles_LIED, dcs_LIED, error,  linestyle='', marker="o", capsize=4, color="blue", markersize=7, label='LIED')
    if has_theory:
        #### Load CED (convential electron diffraction) and compare their diffraction crossection data (from NIST) with LIED ####
        if theory_path is None:
            theory_path = f"DCS_{TARGET}_{int(E0f)}_ev.csv"
        ang_ced, dcs_ced = np.loadtxt(theory_path, delimiter=',', unpack=True, skiprows=10)
        dcs_ced = dcs_ced / np.max(dcs_ced[-5:])#only maximize the around angle 180 deg
        ax.plot(ang_ced, dcs_ced+theory_y_offset, color='black', label=thory_label, lw=2)
    else:
        dcs_ced = (1,) # dummy variable for ylim

    plt.legend()
    if xlim is None:
        xlim = (min(scattering_angles_LIED)-5, 190)
    if ylim is None:
        ylim = (min(min(dcs_LIED), min(dcs_ced))/2, 20)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Differential Elastic-Scattering Cross Section
    target_label = TARGET.replace("_", " ")
    plt.title(f"{target_label} at {int(E0f)}eV ±{round(delta*100)}% | {WL}um | {I0}TW/cm2 | Up = {round(Up)}eV", fontsize=12, fontweight='bold')
    ax.set_xlabel('Scattering angle [deg]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Differential Cross-Section [arb. unit]', fontsize=11, fontweight='bold')
    plt.yscale('log')

    plt.tight_layout()
    if SAVE:
        fig.savefig(f"DCS_{TARGET}_{int(E0f)}eV_{WL}um_{I0}TW_Up{round(Up)}ev_±{round(delta*100)}%.png", dpi=300)
        # Save calculations in a file:
        with open(f"DCS_Calculations_{TARGET}_{int(E0f)}eV_{WL}um_{I0}TW_Up{round(Up)}ev.txt", 'w') as f:
            # store input
            for key in calculations_dict:
                f.write(f"{key} = {calculations_dict[key]}\n")
            f.write(f"\n\nInputs: \nEnergy = {E}eV\nUp = {Up}eV\ndelta = {delta}\nt0 = {t0}\nL = {L}\nE0f = {E0f}\nWL = {WL}\nI0 = {I0}\nTARGET = {TARGET}")
    else:
        # print calculations_dict
        for key in calculations_dict:
            print(key, calculations_dict[key])
        plt.show()



# the update in 2 is that it plots multiple targets at the same figure
def plot_dcs2(E, Up, delta, t0, L, E0f, WL, I0, TARGET, SAVE,\
             time_arr_path=None, has_theory=False, thory_label='CED', theory_path=None,\
             ylim=None, xlim=None, trajectroy_type='long', data_dir_path=r'.', save_path=None,\
            fig=None, ax=None, title=None, data_label='LIED', data_color='blue', show=1, save_calculations=0,
            line_style='', line_width=2, theory_y_offset=0):

    born_phase = search_traj_table("born_phase", "ke_scatter", E/Up, trajectroy_type=trajectroy_type)
    print(f"Born phase = {born_phase}")
    # Check if the born phase is within the long trajectories
    if born_phase > LONG_SHORT_TRAJ_BORN_PHASE_BOUNDARY: # Look at Cosmin thesis page 21 Fig.1.4
        print(f"\n!!Warning!!\nBorn phase = {born_phase} is NOT within the long trajectories\n")
    # Read 2d array from file where the columns are the angle and the rows are the time of flight bins
    path = f"{TARGET}_total_spectra_TDC2228A.dat"
    path = os.path.join(data_dir_path, path)
    angle_header = read_header(path)[0] ## Angle (Columns): 20 to 40 step 0.2
    first_angle, last_angle, angle_step = [int(ang) for ang in re.findall(r'(-?\d+.?\d?)', angle_header)]
    total_spectra_arr = np.loadtxt(path, delimiter='\t')
    # divide each spectrum by the number of laser shots
    path = f"{TARGET}_averaged_count_TDC2228A.dat"
    path = os.path.join(data_dir_path, path)
    angles, avg_count, tot_laser_shots = np.loadtxt(path).T
    #*WARNING*  make sure that angles matches the count, and angles are in ascending order!
    total_spectra_arr /= tot_laser_shots[None, :]


    # dt = 0.151E-9 # time bin width for TDC2228A
    if time_arr_path is None:
        time_arr_path = f"{TARGET}_{WL}um_Round1_ang00_TDC2228A.csv"
        time_arr_path = os.path.join(data_dir_path, time_arr_path)
    time_arr = np.loadtxt(time_arr_path, delimiter=',', usecols=0) *1e-9 # all files have the same time axis
    p = time_to_momentum_axis(time_arr, L=L, t0=t0) # magnitude of momentum in a.u.

    # Loop over Angles and get parrallel and perpendicular momentum for each angle
    angles = np.arange(first_angle, last_angle+angle_step, angle_step)
    dcs_LIED, scattering_angles_LIED, calculations_dict = find_crossection(total_spectra_arr, p, angles, born_phase, Up, delta, trajectroy_type=trajectroy_type)
    # print(dcs_LIED)
    # normalize
    dcs_LIED /= np.mean(dcs_LIED[:4]) #was: max(dcs_LIED[:4])
    error = 1 / np.sqrt(dcs_LIED *tot_laser_shots[None, :])
    ### ------------------- Plotting ------------------- ###
    if fig is None:
        fig, ax = plt.subplots()
    ax.errorbar(scattering_angles_LIED, dcs_LIED, error,  linestyle=line_style, lw=line_width, marker="o", capsize=4, color=data_color, markersize=7, label=data_label)
    if has_theory:
        #### Load CED (convential electron diffraction) and compare their diffraction crossection data (from NIST) with LIED ####
        if theory_path is None:
            theory_path = f"DCS_{TARGET}_{int(E0f)}_ev.csv"
        theory_path = os.path.join(data_dir_path, theory_path)
        ang_ced, dcs_ced = np.loadtxt(theory_path, delimiter=',', unpack=True, skiprows=10)
        dcs_ced = dcs_ced / np.max(dcs_ced[-5:])#only maximize the around angle 180 deg
        ax.plot(ang_ced, dcs_ced+theory_y_offset, color='black', label=thory_label, lw=2)
    else:
        dcs_ced = (1,) # dummy variable for ylim

    plt.legend()
    if xlim is None:
        xlim = (min(scattering_angles_LIED)-5, 190)
    if ylim is None:
        ylim = (min(min(dcs_LIED), min(dcs_ced))/2, 20)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Differential Elastic-Scattering Cross Section
    if title is None:
        target_label = TARGET.replace("_", " ")
        title = f"{target_label} at {int(E0f)}eV ±{round(delta*100)}% | {WL}um | {I0}TW/cm2 | Up = {round(Up)}eV"
    plt.title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Scattering angle [deg]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Differential Cross-Section [arb. unit]', fontsize=11, fontweight='bold')
    plt.yscale('log')

    plt.tight_layout()
    if SAVE:
        if save_path is None:
            save_path = f'{TARGET}_{int(E0f)}eV_{WL}um_{I0}TW_Up{round(Up)}ev'
        fig.savefig(f"DCS_{save_path}.png", dpi=300)
        # Save calculations in a file:
    if save_calculations:
        with open(f"DCS_Calculations_{save_path}.txt", 'a+') as f:
            f.write(f"\nTARGET: {TARGET}\nEnergy = {E}eV\nUp = {Up}eV\ndelta = {delta}\nt0 = {t0}\nL = {L}\nE0f = {E0f}\nWL = {WL}\nI0 = {I0}\n")
            # store input
            for key in calculations_dict:
                f.write(f"{key} = {calculations_dict[key]}\n")
    elif show:
        # print calculations_dict
        plt.show()
    else:
        pass # this keep fig, and ax for further plotting
    for key in calculations_dict:
        print(key, calculations_dict[key])
    return fig, ax


# The update in 3 is:
# 1. It uses the new find_crossection (2 and 3) functions that does not use the jacobian as in find_cross_section (original)
# but it uses summation over the time of flight bins. Also add the interpolation option.
# 2. you can have different normalization types: avg, max, none
# 3. you can have unified number of laser shots for different targets. Options: [numb, None or 1 which means avg hits per shot]
def plot_dcs3(E, Up, delta, t0, L, E0f, WL, I0, TARGET, SAVE,\
             time_arr_path=None, has_theory=False, thory_label='CED', theory_path=None, interpolation=False,\
             ylim=None, xlim=None, trajectroy_type='long', data_dir_path=r'.', save_path=None, calc_momentum_transfer=False,\
            fig=None, ax=None, title=None, data_label='LIED', data_color='blue', show=1, save_calculations=0,
            line_style='', line_width=2, theory_y_offset=0, normalization_type:str='avg', unified_num_of_laser_shots=600e3):
    if interpolation:
        find_crossection_func = find_crossection3 # uses interpolation in the yield_tof
    else:
        find_crossection_func = find_crossection2 # uses the yield_tof as is.
    born_phase = search_traj_table("born_phase", "ke_scatter", E/Up, trajectroy_type=trajectroy_type)
    print(f"Born phase = {born_phase}")
    # Check if the born phase is within the long trajectories
    if born_phase > LONG_SHORT_TRAJ_BORN_PHASE_BOUNDARY: # Look at Cosmin thesis page 21 Fig.1.4
        print(f"\n!!Warning!!\nBorn phase = {born_phase} is NOT within the long trajectories\n")
    # Read 2d array from file where the columns are the angle and the rows are the time of flight bins
    path = f"{TARGET}_total_spectra_TDC2228A.dat"
    path = os.path.join(data_dir_path, path)
    angle_header = read_header(path)[0] ## Angle (Columns): 20 to 40 step 0.2
    first_angle, last_angle, angle_step = [int(ang) for ang in re.findall(r'(-?\d+.?\d?)', angle_header)]
    total_spectra_arr = np.loadtxt(path, delimiter='\t')
    # divide each spectrum by the number of laser shots
    path = f"{TARGET}_averaged_count_TDC2228A.dat"
    path = os.path.join(data_dir_path, path)
    angles, avg_count, tot_laser_shots = np.loadtxt(path).T
    #*WARNING*  make sure that angles matches the count, and angles are in ascending order!
    if unified_num_of_laser_shots is None:
        # Warning: Unified number of laser shots should be provieded when comparing different targets
        unified_num_of_laser_shots = np.mean(tot_laser_shots)
    total_spectra_arr = total_spectra_arr * unified_num_of_laser_shots / tot_laser_shots[None, :]

    # dt = 0.151E-9 # time bin width for TDC2228A
    if time_arr_path is None:
        time_arr_path = f"{TARGET}_{WL}um_Round1_ang00_TDC2228A.csv"
        time_arr_path = os.path.join(data_dir_path, time_arr_path)
    time_arr = np.loadtxt(time_arr_path, delimiter=',', usecols=0)# nanoseconds
    # Loop over Angles and get parrallel and perpendicular momentum for each angle
    angles = np.arange(first_angle, last_angle+angle_step, angle_step)
    dcs_LIED, scattering_angles_LIED, calculations_dict = find_crossection_func(total_spectra_arr, time_arr, angles, born_phase, Up, t0, L, delta, trajectroy_type=trajectroy_type)
    # print(dcs_LIED)
    error = 1 / np.sqrt(dcs_LIED)
    # normalize
    if normalization_type == 'avg':
        norm_factor = np.mean(dcs_LIED[:4])
    elif normalization_type == 'max':
        norm_factor = max(dcs_LIED[:4])
    elif normalization_type == 'none':
        norm_factor = 1
    dcs_LIED /= norm_factor
    error /= norm_factor
    ### ------------------- Plotting ------------------- ###
    if fig is None:
        # For DCS
        fig, ax = plt.subplots()
        # For Momentum Transfer
        # fig2, ax2 = plt.subplots()
    ax.errorbar(scattering_angles_LIED, dcs_LIED, error,  linestyle=line_style, lw=line_width, marker="o", capsize=4, color=data_color, markersize=7, label=data_label)

    if has_theory:
        #### Load CED (convential electron diffraction) and compare their diffraction crossection data (from NIST) with LIED ####
        if theory_path is None:
            theory_path = f"DCS_{TARGET}_{int(E0f)}_ev.csv"
        theory_path = os.path.join(data_dir_path, theory_path)
        ang_ced, dcs_ced = np.loadtxt(theory_path, delimiter=',', unpack=True, skiprows=10)
        dcs_ced = dcs_ced / np.max(dcs_ced[-5:])#only maximize the around angle 180 deg
        ax.plot(ang_ced, dcs_ced+theory_y_offset, color='black', label=thory_label, lw=2)
    else:
        dcs_ced = (1,) # dummy variable for ylim

    plt.legend()
    if xlim is None:
        xlim = (min(scattering_angles_LIED)-5, 190)
    if ylim is None:
        ylim = (min(min(dcs_LIED), min(dcs_ced))/2, 20)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Differential Elastic-Scattering Cross Section
    if title is None:
        target_label = TARGET.replace("_", " ")
        title = f"{target_label} at {int(E0f)}eV ±{round(delta*100)}% | {WL}um | {I0}TW/cm2 | Up = {round(Up)}eV"
    plt.title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Scattering angle [deg]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Differential Cross-Section [arb. unit]', fontsize=11, fontweight='bold')
    plt.yscale('log')

    plt.tight_layout()
    if SAVE:
        if save_path is None:
            save_path = f'{TARGET}_{int(E0f)}eV_{WL}um_{I0}TW_Up{round(Up)}ev'
        fig.savefig(f"DCS_{save_path}.png", dpi=300)
        # Save calculations in a file:
    if save_calculations:
        with open(f"DCS_Calculations_{save_path}.txt", 'a+') as f:
            f.write(f"\nTARGET: {TARGET}\nEnergy = {E}eV\nUp = {Up}eV\ndelta = {delta}\nt0 = {t0}\nL = {L}\nE0f = {E0f}\nWL = {WL}\nI0 = {I0}\n")
            # store input
            for key in calculations_dict:
                f.write(f"{key} = {calculations_dict[key]}\n")
    elif show:
        # print calculations_dict
        plt.show()
    else:
        pass # this keep fig, and ax for further plotting
    for key in calculations_dict:
        print(key, calculations_dict[key])

    # Momentum Transfer
    if calc_momentum_transfer:
        # Find momentum transfer
        # q = 2 p_r sin(phi/2)
        pr = calculations_dict['Pr [au]']
        momentum_transfer = 2 * pr * np.sin(np.deg2rad(scattering_angles_LIED)/2)
        fig2, ax2 = plt.subplots()
        ax2.plot(momentum_transfer, dcs_LIED, 'o-', color="blue")
        ax2.set_xlabel('Momentum Transfer [au]', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Differential Cross-Section [arb. unit]', fontsize=11, fontweight='bold')
        ax2.set_yscale('log')
        plt.show()

    return fig, ax



###########################

