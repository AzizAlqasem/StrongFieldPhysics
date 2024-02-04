import os
import numpy as np
from StrongFieldPhysics.classical.born_return_times import get_born_return_times
from StrongFieldPhysics.calculations.atomic_unit import angluar_freq_au
from StrongFieldPhysics.calculations.constants import LONG_SHORT_TRAJ_BORN_PHASE_BOUNDARY
from StrongFieldPhysics.tools.arrays import find_index_near

LIED_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

def vector_potential(t, w, A0=1, phase=0):
    return A0 * np.cos(w * t + phase)

def vector_potential_rescatter(ke_scatter, ke_backscatter):
    # energy in atomic units
    return np.sqrt(2 * ke_backscatter) - np.sqrt(2 * ke_scatter)

def ke_scatter(t_born, t_return, w, Up=1, phase=0):
    A0=1 # Actually Up = A0^2/4, so using Up is enough
    A_born = vector_potential(t_born, w, A0, phase)
    A_return = vector_potential(t_return, w, A0, phase)
    return 2 * Up * (A_return - A_born)**2

def ke_backscatter(t_born, t_return, w, Up=1, phase=0):
    A0=1 # Actually Up = A0^2/4, so using Up is enough
    A_born = vector_potential(t_born, w, A0, phase)
    A_return = vector_potential(t_return, w, A0, phase)
    return 2 * Up * (2*A_return - A_born)**2

def momentum_scatter_amp(t_born, t_return, w, A0=1, phase=0):
    # Only the amplitude of the momentum scattering
    A_born = vector_potential(t_born, w, A0, phase)
    A_return = vector_potential(t_return, w, A0, phase)
    return A_born - A_return

def momentum_rescatter_amp(t_born, t_return, w, A0=1, phase=0):
    # Only the amplitude of the momentum
    # P_rescatter = - P_scatter
    return -momentum_scatter_amp(t_born, t_return, w, A0, phase)

def momentum_rescatter_amp_ke(ke_scatter):
    return np.sqrt(2 * ke_scatter)

def momentum_backscatter_amp(t_born, t_return, w, A0=1, phase=0):
    # Only the amplitude of the momentum scattering
    # This is same as measured momentum along the laser pol direction
    A_born = vector_potential(t_born, w, A0, phase)
    A_return = vector_potential(t_return, w, A0, phase)
    return 2*A_return - A_born

######## Angles
def detected_momentum_at_scattring_angle(ar, pr, scatter_angle_deg):
    # detected momentum = sqrt (A_r**2 + p_r**2 - 2*A_r*p_r*cos(scatter_angle))
    # ar: vector potential at rescattering
    # pr: rescattering momentum
    scatter_angle = np.deg2rad(scatter_angle_deg)
    return np.sqrt(ar**2 + pr**2 - 2*ar*pr*np.cos(scatter_angle))

def detected_momentum_at_detected_angle(ar, pr, detected_angle_deg):
    # detected momentum = sqrt (A_r**2 + p_r**2 - 2*A_r*p_r*cos(scatter_angle))
    # ar: vector potential at rescattering
    # pr: rescattering momentum
    scattering_angle_deg = scattering_angle(detected_angle_deg, ar, pr)
    return detected_momentum_at_scattring_angle(ar, pr, scattering_angle_deg)


def scattering_angle(detected_angle_deg, ar, pr):# this is the correct way to find the scattering angle
    # detected_angle: angle between pd and laser polarization axis (experimental)
    # ar: vector potential at rescattering
    # pr: rescattering momentum
    da = np.deg2rad(detected_angle_deg)
    sa =  np.pi - da - np.arcsin(ar/pr * np.sin(da))
    return np.rad2deg(sa)


##### Do not use the following functions, they are not correct
# def scattering_angle(pd, pr, detected_angle_deg)!: # does not work with det_ang=0 (it gives 0 which is wrong, should be 180)
#     # pd: detected momentum
#     # pr: rescattering momentum
#     # detected_angle: angle between pd and laser polarization axis
#     detected_angle = np.deg2rad(detected_angle_deg)
#     sa = np.arcsin( pd/pr * np.sin(detected_angle) )
#     return np.rad2deg(sa)

# def detected_angle(pd, pr, scattering_angle_deg)!: # does not work with scat_ang=0 (it gives 0 which is wrong, should be 180)
#     # pd: detected momentum
#     # pr: rescattering momentum
#     scattering_angle = np.deg2rad(scattering_angle_deg)
#     da = np.arcsin(pr/pd * np.sin(scattering_angle))
#     return np.rad2deg(da)
######


######## Trajectories Table
# Generate table
def generate_trajectories_table(dt_au=0.01, wavelength=3000, max_n_cyckes=20, min_t_diff=1, path=None):
    # expensive funtion to run
    tb,tr = get_born_return_times(wavelength_nm=wavelength, max_n_cycles=max_n_cyckes, dt_au=dt_au, min_t_diff=min_t_diff)
    w = angluar_freq_au(wavelength)
    cond = np.isnan(tr)==False
    # calculations (explicitly written for clarity)
    KE_res = 2 * (np.cos(w*tr) - np.cos(w*tb))**2
    KE_backres = 2 * (2*np.cos(w*tr) - np.cos(w*tb))**2
    A_bir = np.cos(w*tb)
    A_res = np.cos(w*tr)
    P_scat_0deg = A_bir - A_res
    P_backscatter_0deg = 2*A_res - A_bir
    # save
    if path is None:
        path = f'{LIED_DIR_PATH}/LIED_table.dat'
    np.savetxt(path, np.array([tb*w, tr*w, KE_res, KE_backres, A_bir, A_res, P_scat_0deg, P_backscatter_0deg]).T, header="All data corresponds to the unified Phase=Time*angfreq. KE is unit of Up. A and P are unit of A0 and all are at the polarization axis. dt=0.01au\nBornPhase\tReturnPhaser\tKE_scatter\tKE_backscatter\tA_birth\tA_scatter\tP_scatter\tP_backscatter")
    path = path.replace("table","table2")
    np.savetxt(path, np.array([tb[cond]*w, tr[cond]*w, KE_res[cond], KE_backres[cond], A_bir[cond], A_res[cond], P_scat_0deg[cond], P_backscatter_0deg[cond]]).T, header="Only Scattering events that has return times\nAll data corresponds to the unified Phase=Time*angfreq. KE is unit of Up. A and P are unit of A0 and all are at the polarization axis. dt=0.01au\nBornPhase\tReturnPhaser\tKE_scatter\tKE_backscatter\tA_birth\tA_scatter\tP_scatter\tP_backscatter")

# Load table
def load_trajectories_table_all(db=2, path=None):
    #tb,tr,KE_res,KE_backres,A_bir,A_res,P_res_0deg,P_backscatter_0deg
    # get path of this file
    if path is None:
        path = f'{LIED_DIR_PATH}/LIED_table{db}.dat'
    return np.loadtxt(path, unpack=True)
    # return np.loadtxt(f'LIED_table{db}.dat', unpack=True)

def trajectories_search_table(db=2, path=None): # Closure function
    # !!Warning!! the following search is not accurate. Except when trajectory_type="long"
    # search("born_phase", "ke_scatter", 1) # there are two born phases for the same ke_scatter (and other quantities)
    data = load_trajectories_table_all(db=db, path=path)
    columns = ['born_phase', 'return_phase', 'ke_scatter', 'ke_backscatter', 'a_birth', 'a_scatter', 'p_scatter', 'p_backscatter_0deg']
    # find the  border of long and short trajectories in terms of born phase
    boundry_indx = find_index_near(data[columns.index("born_phase")], LONG_SHORT_TRAJ_BORN_PHASE_BOUNDARY)
    # this born phase has only one unique ke_scatter, and separates the long and short trajectories
    def get_data(target_column:str, search_by:str, search_value:float, trajectroy_type="long"):
        target_column = target_column.lower()
        search_by = search_by.lower()
        assert target_column in columns, f"target_column must be one of {columns}"
        assert search_by in columns, f"search_by must be one of {columns}"
        target_column_index = columns.index(target_column)
        search_by_index = columns.index(search_by)
        if trajectroy_type == "long":
            searched_arr = data[search_by_index][:boundry_indx]
            offset = 0
        elif trajectroy_type == "short":
            searched_arr = data[search_by_index][boundry_indx:]
            offset = boundry_indx
        else:
            searched_arr = data[search_by_index]
        return data[target_column_index][find_index_near(searched_arr, search_value)+offset]
    return get_data





####################### Draft ############################
# def detected_momentum_at_angle(scatter_angle=0, **kwargs):
#     # kwargs are required
#     # detected momentum = sqrt (A_r**2 + p_r**2 - 2*A_r*p_r*cos(scatter_angle))
#     # if A_r or p_r are not given in the kwargs, they will be calculated from the other kwargs
#     assert len(kwargs) > 1, "kwargs are required"