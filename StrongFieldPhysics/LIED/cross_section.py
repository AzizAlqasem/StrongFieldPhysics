import numpy as np
from scipy.interpolate import interp1d

from StrongFieldPhysics.classical.trajectories import trajectories_search_table, vector_potential_rescatter, momentum_rescatter_amp_ke\
    , detected_momentum_at_detected_angle, scattering_angle, detected_momentum_at_scattring_angle
from StrongFieldPhysics.calculations.atomic_unit import energy_ev_to_au
from StrongFieldPhysics.tools.arrays import find_index_near
from StrongFieldPhysics.time_of_flight.tof_to_momentum import momentum_au_to_tof_ns


# Load table
#['born_phase', 'return_phase', 'ke_scatter', 'ke_backscatter', 'a_birth', 'a_scatter', 'p_scatter', 'p_backscatter_0deg']
search_traj_table = trajectories_search_table(path=r"C:\Users\alqasem.2\OneDrive - The Ohio State University\Strong Field Squad\Alkali Machine and VMI\Code\StrongFieldPhysics\StrongFieldPhysics\classical\LIED_table2.dat") # function




def find_crossection(yield_t, p_arr, det_angles, born_phase, up_ev, delta=0.02, trajectroy_type='long'):
    """
    yield_t: 2d array of yield as a function TOF. Columns represent different angles and rows represent TOF bins.
    p_arr: detected (experimantl) momentum array
    det_angles: detected angle array in degrees (laser polarization angle with respect to the time of flight axis)
    born_phase: born phase of the electron (1.6 to 1.9 are for long trajectories)
    up_ev: ponderomotive energy [ev]
    delta: delta=(dp/p), uncertainty in momentum. values from 0 to 1
    """
    # From the born phase, get the ke_scatter and ke_backscatter
    ke_scatter = search_traj_table("ke_scatter", "born_phase", born_phase, trajectroy_type=trajectroy_type)
    ke_backscatter = search_traj_table("ke_backscatter", "born_phase", born_phase, trajectroy_type=trajectroy_type)
    # convert Up to au
    up_au = energy_ev_to_au(up_ev)
    ke_scatter = ke_scatter * up_au # convert from units of up to au
    ke_backscatter = ke_backscatter * up_au
    # Find limit of the uncertainty in energy provided as delta
    ke_scatter_min = ke_scatter * (1 - delta)
    ke_scatter_max = ke_scatter * (1 + delta)
    ke_backscatter_min = ke_backscatter * (1 - delta)
    ke_backscatter_max = ke_backscatter * (1 + delta)
    # From the ke_scatter and ke_backscatter, get the vector potential and momentum at rescattering
    ar = vector_potential_rescatter(ke_scatter, ke_backscatter)
    pr = momentum_rescatter_amp_ke(ke_scatter)
    ar_min = vector_potential_rescatter(ke_scatter_min, ke_backscatter_min)
    pr_min = momentum_rescatter_amp_ke(ke_scatter_min)
    ar_max = vector_potential_rescatter(ke_scatter_max, ke_backscatter_max)
    pr_max = momentum_rescatter_amp_ke(ke_scatter_max)
    # Loop over the detected angle
    yield_sum = np.zeros_like(det_angles, dtype=np.float64)
    scatter_angle_deg_arr = np.zeros_like(det_angles, dtype=np.float64)
    for i, det_angle in enumerate(det_angles):
        # From the ar and pr, get the detected momentum at the detected angle
        # pd = detected_momentum_at_detected_angle(ar, pr, det_angle)
        pd_min = detected_momentum_at_detected_angle(ar_min, pr_min, det_angle)
        pd_max = detected_momentum_at_detected_angle(ar_max, pr_max, det_angle)
        # print("Detected angle: ", det_angle)
        # print("Ke Range [Up] ", pd_min**2/(2*up_au), pd_max**2/(2*up_au))
        if pd_min < p_arr[0] or pd_max > p_arr[-1]:
            print("Warning: The edge of the momentum array is reached!")
            continue
        # Find the coressponding yield between pd1 and pd2 in the data (after applying the jacobian for momentum)
        indx_min = find_index_near(p_arr, pd_min)
        indx_max = find_index_near(p_arr, pd_max)
        spectrum = yield_t[:, i]
        # Apply the jacobian
        spectrum = spectrum[::-1] / p_arr**2
        yield_sum[i] = np.sum(spectrum[indx_min:indx_max]) # assuming that angle i corresponds to the same index in the yield_t
        scatter_angle_deg_arr[i] = scattering_angle(det_angle, ar, pr) # degree

    # Store important calculations in a dictionary
    calc_dict = {"KE Scattering [Up]": ke_scatter/up_au, \
                 "KE BackScattering [Up]": ke_backscatter/up_au, \
                 "Pr [au]": pr, \
                 "Ar [au]": ar, \
                 "Uncertainty [%]": round(delta*100,1),
                 "Born Phase [rad]": born_phase}

    return yield_sum, scatter_angle_deg_arr, calc_dict


def find_crossection2(yield_t, tof_arr, det_angles, born_phase, up_ev, t0, L, delta=0.02, trajectroy_type='long'):
    """
    yield_t: 2d array of yield as a function TOF. Columns represent different angles and rows represent TOF bins.
    tof_arr: time of flight array from the raw data in ns. not calibrated. t0 and L will calibrate it.
    det_angles: detected angle array in degrees (laser polarization angle with respect to the time of flight axis)
    born_phase: born phase of the electron (1.6 to 1.9 are for long trajectories)
    up_ev: ponderomotive energy [ev]
    t0: time zero of the experiment in seconds
    L: Time of flight length in meters
    delta: delta=(dp/p), uncertainty in momentum. values from 0 to 1
    """
    # From the born phase, get the ke_scatter and ke_backscatter
    ke_scatter = search_traj_table("ke_scatter", "born_phase", born_phase, trajectroy_type=trajectroy_type)
    ke_backscatter = search_traj_table("ke_backscatter", "born_phase", born_phase, trajectroy_type=trajectroy_type)
    # convert Up to au
    up_au = energy_ev_to_au(up_ev)
    ke_scatter = ke_scatter * up_au # convert from units of up to au
    ke_backscatter = ke_backscatter * up_au
    # Find limit of the uncertainty in energy provided as delta
    ke_scatter_min = ke_scatter * (1 - delta)
    ke_scatter_max = ke_scatter * (1 + delta)
    ke_backscatter_min = ke_backscatter * (1 - delta)
    ke_backscatter_max = ke_backscatter * (1 + delta)
    # From the ke_scatter and ke_backscatter, get the vector potential and momentum at rescattering
    ar = vector_potential_rescatter(ke_scatter, ke_backscatter)
    pr = momentum_rescatter_amp_ke(ke_scatter)
    ar_min = vector_potential_rescatter(ke_scatter_min, ke_backscatter_min)
    pr_min = momentum_rescatter_amp_ke(ke_scatter_min)
    ar_max = vector_potential_rescatter(ke_scatter_max, ke_backscatter_max)
    pr_max = momentum_rescatter_amp_ke(ke_scatter_max)
    # Loop over the detected angle
    yield_sum = np.zeros_like(det_angles, dtype=np.float64)
    scatter_angle_deg_arr = np.zeros_like(det_angles, dtype=np.float64)
    for i, det_angle in enumerate(det_angles):
        # From the ar and pr, get the detected momentum at the detected angle
        # pd = detected_momentum_at_detected_angle(ar, pr, det_angle)
        pd_min = detected_momentum_at_detected_angle(ar_min, pr_min, det_angle)
        pd_max = detected_momentum_at_detected_angle(ar_max, pr_max, det_angle)
        # Note: the larger the momentum the smaller the tof
        tof_max = momentum_au_to_tof_ns(pd_min, t0_ns=t0*1e9, L=L) # ns
        tof_min = momentum_au_to_tof_ns(pd_max, t0_ns=t0*1e9, L=L) # ns
        if tof_min < tof_arr[0] or tof_max > tof_arr[-1]:
            print("Warning: The edge of the TOF array is reached!")
            continue
        # Find the coressponding yield between tof_min and tof_max in the data (without applying the jacobian)
        indx_min = find_index_near(tof_arr, tof_min)
        indx_max = find_index_near(tof_arr, tof_max)
        yield_sum[i] = np.sum(yield_t[indx_min:indx_max+1, i]) # assuming that angle i corresponds to the same index in the yield_t
        scatter_angle_deg_arr[i] = scattering_angle(det_angle, ar, pr) # degree

    # Store important calculations in a dictionary
    calc_dict = {"KE Scattering [Up]": ke_scatter/up_au, \
                 "KE BackScattering [Up]": ke_backscatter/up_au, \
                 "Pr [au]": pr, \
                 "Ar [au]": ar, \
                 "Uncertainty [%]": round(delta*100,1),
                 "Born Phase [rad]": born_phase}

    return yield_sum, scatter_angle_deg_arr, calc_dict


def find_crossection3(yield_t, tof_arr, det_angles, born_phase, up_ev, t0, L, delta=0.02, trajectroy_type='long'):
    """
    yield_t: 2d array of yield as a function TOF. Columns represent different angles and rows represent TOF bins.
    tof_arr: time of flight array from the raw data in ns. not calibrated. t0 and L will calibrate it.
    det_angles: detected angle array in degrees (laser polarization angle with respect to the time of flight axis)
    born_phase: born phase of the electron (1.6 to 1.9 are for long trajectories)
    up_ev: ponderomotive energy [ev]
    t0: time zero of the experiment in seconds
    L: Time of flight length in meters
    delta: delta=(dp/p), uncertainty in momentum. values from 0 to 1
    """
    # From the born phase, get the ke_scatter and ke_backscatter
    ke_scatter = search_traj_table("ke_scatter", "born_phase", born_phase, trajectroy_type=trajectroy_type)
    ke_backscatter = search_traj_table("ke_backscatter", "born_phase", born_phase, trajectroy_type=trajectroy_type)
    # convert Up to au
    up_au = energy_ev_to_au(up_ev)
    ke_scatter = ke_scatter * up_au # convert from units of up to au
    ke_backscatter = ke_backscatter * up_au
    # Find limit of the uncertainty in energy provided as delta
    ke_scatter_min = ke_scatter * (1 - delta)
    ke_scatter_max = ke_scatter * (1 + delta)
    ke_backscatter_min = ke_backscatter * (1 - delta)
    ke_backscatter_max = ke_backscatter * (1 + delta)
    # From the ke_scatter and ke_backscatter, get the vector potential and momentum at rescattering
    ar = vector_potential_rescatter(ke_scatter, ke_backscatter)
    pr = momentum_rescatter_amp_ke(ke_scatter)
    ar_min = vector_potential_rescatter(ke_scatter_min, ke_backscatter_min)
    pr_min = momentum_rescatter_amp_ke(ke_scatter_min)
    ar_max = vector_potential_rescatter(ke_scatter_max, ke_backscatter_max)
    pr_max = momentum_rescatter_amp_ke(ke_scatter_max)
    # Loop over the detected angle
    yield_sum = np.zeros_like(det_angles, dtype=np.float64)
    scatter_angle_deg_arr = np.zeros_like(det_angles, dtype=np.float64)
    tof_arr_interp = np.linspace(tof_arr[0], tof_arr[-1], len(tof_arr)*10)
    # yield_t_interp = np.zeros((len(tof_arr_interp), len(det_angles)), dtype=np.float64)
    for i, det_angle in enumerate(det_angles):
        # From the ar and pr, get the detected momentum at the detected angle
        # pd = detected_momentum_at_detected_angle(ar, pr, det_angle)
        scattering_ang_min = scattering_angle(det_angle, ar_min, pr_min) # degree
        scattering_ang_max = scattering_angle(det_angle, ar_max, pr_max)
        pd_min = detected_momentum_at_scattring_angle(ar_min, pr_min, scattering_ang_min)
        pd_max = detected_momentum_at_scattring_angle(ar_max, pr_max, scattering_ang_max)
        # Note: the larger the momentum the smaller the tof
        tof_max = momentum_au_to_tof_ns(pd_min, t0_ns=t0*1e9, L=L) # ns
        tof_min = momentum_au_to_tof_ns(pd_max, t0_ns=t0*1e9, L=L) # ns
        if tof_min < tof_arr[0] or tof_max > tof_arr[-1]:
            print("Warning: The edge of the TOF array is reached!")
            continue
        # Find the coressponding yield between tof_min and tof_max in the data (without applying the jacobian)
        indx_min = find_index_near(tof_arr_interp, tof_min)
        indx_max = find_index_near(tof_arr_interp, tof_max)
        # interpolate the spectrum
        spline = interp1d(tof_arr, yield_t[:, i], kind='cubic')
        spectrum_interp = spline(tof_arr_interp)
        yield_sum[i] = np.sum(spectrum_interp[indx_min:indx_max+1]) # assuming that angle i corresponds to the same index in the yield_t
        scatter_angle_deg_arr[i] = (scattering_ang_min + scattering_ang_max) / 2.0  #scattering_angle(det_angle, ar, pr) # degree

    # Store important calculations in a dictionary
    calc_dict = {"KE Scattering [Up]": ke_scatter/up_au, \
                 "KE BackScattering [Up]": ke_backscatter/up_au, \
                 "Pr [au]": pr, \
                 "Ar [au]": ar, \
                 "Uncertainty [%]": round(delta*100,1),
                 "Born Phase [rad]": born_phase}

    return yield_sum, scatter_angle_deg_arr, calc_dict

# Wrong!
# def find_crossection(yield_p, p_arr, det_angles, born_phase, up_ev, delta=0.02):
#     """
#     yield_p: 2d array of yield as a function of detected (experimantal) momentum and detected angle
#     p_arr: detected (experimantl) momentum array
#     det_angles: detected angle array in degrees (laser polarization angle with respect to the time of flight axis)
#     born_phase: born phase of the electron (1.6 to 1.9 are for long trajectories)
#     up_ev: ponderomotive energy [ev]
#     delta: delta=(dp/p), uncertainty in momentum. values from 0 to 1
#     """
#     # From the born phase, get the ke_scatter and ke_backscatter
#     ke_scatter = search_traj_table("ke_scatter", "born_phase", born_phase)
#     ke_backscatter = search_traj_table("ke_backscatter", "born_phase", born_phase)
#     # convert Up to au
#     up_au = energy_ev_to_au(up_ev)
#     ke_scatter = ke_scatter * up_au # convert from units of up to au
#     ke_backscatter = ke_backscatter * up_au
#     # From the ke_scatter and ke_backscatter, get the vector potential and momentum at rescattering
#     ar = vector_potential_rescatter(ke_scatter, ke_backscatter) # Fixed value for the cross section
#     pr = momentum_rescatter_amp_ke(ke_scatter) # Fixed value for the cross section
#     # Loop over the detected angle
#     yield_sum = np.zeros_like(det_angles, dtype=np.float64)
#     scatter_angle_deg_arr = np.zeros_like(det_angles, dtype=np.float64)
#     for i, det_angle in enumerate(det_angles):
#         # From the ar and pr, get the detected momentum at the detected angle
#         pd = detected_momentum_at_detected_angle(ar, pr, det_angle)
#         # Find the coressponding yield between pd1 and pd2 in the data (after applying the jacobian for momentum)
#         pd_min = pd * (1 - delta)
#         pd_max = pd * (1 + delta)
#         if pd_min < p_arr[0] or pd_max > p_arr[-1]:
#             print("Warning: The edge of the momentum array is reached!")
#             continue
#         indx_min = find_index_near(p_arr, pd_min)
#         indx_max = find_index_near(p_arr, pd_max)
#         spectrum = yield_p[:, i]
#         # Apply the jacobian
#         spectrum = spectrum[::-1] / p_arr**2 # p_arr is already flipped and in atomic units
#         yield_sum[i] = np.sum(spectrum[indx_min:indx_max]) # assuming that angle i corresponds to the same index in the yield_p
#         scatter_angle_deg_arr[i] = scattering_angle(det_angle, ar, pr) # degree
#     return yield_sum, scatter_angle_deg_arr






















