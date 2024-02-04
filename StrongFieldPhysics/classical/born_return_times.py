# Ionization (born) time and rescattering time
# Run by:
# tb,tr = get_born_return_times(3000, max_n_cycles=20, dt_au = 0.01, min_t_diff=1)
# plt.plot(tb, tr); plt.show()
# w = angluar_freq_au(3000)
# plt.plot(tb*w, np.cos(tr*w)); plt.show()
# tb,tr = get_born_return_times(3000, max_n_cycles=20, dt_au = 0.01, min_t_diff=1)
# np.savetxt('Born_Return_times.dat', np.array([tb*w, tr*w]).T, header="Time*angfreq. dt=0.01au\nBornPhase\tReturnPhaser")
# cond = np.isnan(tr)==False
# np.savetxt('Born_Return_times_only.dat', np.array([tb[cond]*w, tr[cond]*w]).T, header="Time*angfreq. dt=0.01au\nBornPhase\tReturnPhaser")

# KE_res = 2 * (np.cos(w*tr) - np.cos(w*tb))**2
# plt.plot(tb*w, KE_res);plt.show()
# KE_backres = 2 * (2*np.cos(w*tr) - np.cos(w*tb))**2
# plt.plot(tb*w, KE_backres);plt.show()
# A_bir = np.cos(w*tb)
# A_res = np.cos(w*tr)
# plt.plot(w*tb, A_bir);plt.show()
# plt.plot(w*tr, A_res);plt.show()
# P_res_0deg = A_bir - A_res
# plt.plot(P_res_0deg);plt.show()
# P_backscatter_0deg = 2*A_res - A_bir
# np.savetxt('LIED_table.dat', np.array([tb*w, tr*w, KE_res, KE_backres, A_bir, A_res, P_res_0deg, P_backscatter_0deg]).T, header="All data corresponds to the unified Phase=Time*angfreq. KE is unit of Up. A and P are unit of A0 and all are at the polarization axis. dt=0.01au\nBornPhase\tReturnPhaser\tKE_scatter\tKE_backscatter\tA_birth\tA_scatter\tP_scatter\tP_backscatter")
# np.savetxt('LIED_table2.dat', np.array([tb[cond]*w, tr[cond]*w, KE_res[cond], KE_backres[cond], A_bir[cond], A_res[cond], P_res_0deg[cond], P_backscatter_0deg[cond]]).T, header="Only Scattering events that has return times\nAll data corresponds to the unified Phase=Time*angfreq. KE is unit of Up. A and P are unit of A0 and all are at the polarization axis. dt=0.01au\nBornPhase\tReturnPhaser\tKE_scatter\tKE_backscatter\tA_birth\tA_scatter\tP_scatter\tP_backscatter")


from scipy.optimize import root, brentq
import numpy as np
from numba import njit

from StrongFieldPhysics.calculations.atomic_unit import angluar_freq_au

def _born_return_time_func(t_ret, t_born, w):
    # if abs(t_ret - t_born) < 10:
    #     return np.inf # trivial case!
    # time and angular freq in atomic units
    intg = (np.sin(t_ret * w) - np.sin(t_born * w))/ w
    return (intg - (np.cos(t_born * w) * (t_ret - t_born)))#**2

# Works for all phases
def get_born_return_times(wavelength_nm, dt_au=0.1, max_n_cycles=4, min_t_diff=10): # slow function
    w = angluar_freq_au(wavelength_nm)
    max_t_return = (max_n_cycles * 2 * np.pi) / w
    t_born = np.arange(0, 2 * np.pi / w, dt_au, dtype=np.float64)
    t_return = np.zeros_like(t_born, dtype=np.float64)
    for i, tb in enumerate(t_born):
        tr = tb + min_t_diff
        diff = _born_return_time_func(tr, tb, w)
        tr_found = np.nan
        if diff > 0:
            t_return[i] = tr_found
            continue
        # fast search
        while diff < 1 and tr < max_t_return:
            tr += 1 # big steps
            diff = _born_return_time_func(tr, tb, w)
            if diff > 0:
                tr_found = tr
                break
        if np.isnan(tr_found):
            t_return[i] = np.nan
            continue
        # fine search
        tr = brentq(_born_return_time_func, tr_found-2, tr_found, args=(tb, w), xtol=1e-7, rtol=1e-7)
        t_return[i] = tr
    return t_born, t_return


### Old functions that work exept for the smaller born times (phases below 1.8 rad)
# def get_return_time(t_born, wavelength_nm):
#     """Calculates the return time of an electron after ionization
#     Args:
#         t_born (float): Born time in atomic units
#         wavelength_nm (float): laser wavelength in nm
#     Returns:
#         float: return time in atomic units
#     """
#     w = angluar_freq_au(wavelength_nm)
#     # Constrains: t_ret != t_born
#     min = t_born + 1e-5
#     max = (5 * np.pi) / w

#     try:
#         t_ret = brentq(_born_return_time_func, min, max, args=(t_born, w), xtol=1e-7, rtol=1e-7, )
#     except ValueError:
#         return np.nan
#     return t_ret

# def get_return_time_arr(t_born_arr, wavelength_nm):
#     res = np.zeros_like(t_born_arr, dtype=np.float64)
#     for i, t_born in enumerate(t_born_arr):
#         res[i] = get_return_time(t_born, wavelength_nm)
#     return res





