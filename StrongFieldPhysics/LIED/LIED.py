"""This file is a rewritten of cross_section.py and plot_cross_section.py"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re
import os
from scipy.interpolate import interp1d

from StrongFieldPhysics.parser.data_files import read_header
from StrongFieldPhysics.time_of_flight.tof_to_momentum import time_to_momentum_axis
from StrongFieldPhysics.LIED.cross_section import search_traj_table
from StrongFieldPhysics.classical.trajectories import vector_potential_rescatter, momentum_rescatter_amp_ke\
    , scattering_angle, detected_momentum_at_scattring_angle
from StrongFieldPhysics.calculations.atomic_unit import energy_ev_to_au, momentum_au_to_angstrom_inv
from StrongFieldPhysics.tools.arrays import find_index_near
from StrongFieldPhysics.time_of_flight.tof_to_momentum import momentum_au_to_tof_ns
from StrongFieldPhysics.calculations.fourier import fft, filter_signal, correct_phase





class LIED:
    """LIED object that does main LIED calculations such as calculating the cross section and momentum transfa and plotting the results
    """
    N_OF_OBJS = -1
    COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    def __init__(self, E, Up, t0, L, WL, I0, TARGET, data_dir_path=r'.', E0f=None, tab_color=False):
        self.E = E # eV
        self.Up = Up # eV
        self.t0 = t0 # sec
        self.L = L # m
        self.WL = WL # um
        self.I0 = I0 # TW/cm2  #only used for labeling
        self.TARGET = TARGET # string
        self.data_dir_path = data_dir_path
        if E0f is None: # E0f is just the energy we want to show in the plots
            E0f = E # eV
        self.E0f = E0f # eV
        self.id = self._gen_id()

        self.tab_color=tab_color # if True, the color will be "tab:color"


    # handle data
    def load_data(self, fn=None, dir_path=None, median_threshold:int=None): # Load the 2d array where the columns are the angle and the rows are the time of flight bins
        '''Load the 2d array where the columns are the angle and the rows are the time of flight bins
        And create the angles array.
        The time array has to be loaded separately using load_time_arr
        '''
        if fn is None:
            extra = f'_median{median_threshold}' if median_threshold else ''
            fn = f"{self.TARGET}_total_spectra{extra}_TDC2228A.dat"
        if dir_path is None:
            dir_path = self.data_dir_path # current directory
        path = os.path.join(dir_path, fn)
        self.yield_tof_arr_2d = np.loadtxt(path, delimiter='\t') # Yield as a function of time of flight and angle
        # Angles
        angle_header = read_header(path)[0] #
        first_angle, last_angle, angle_step = [int(ang) for ang in re.findall(r'(-?\d+.?\d?)', angle_header)]
        self.detected_angles = np.arange(first_angle, last_angle+angle_step, angle_step)

    def load_time_arr(self, fn=None, dir_path=None): # Load the time array and calculate the momentum array
        if fn is None:
            fn = f"{self.TARGET}_{self.WL}um_Round1_ang00_TDC2228A.csv"
        if dir_path is None:
            dir_path = self.data_dir_path
        path = os.path.join(dir_path, fn)
        self.time_arr = np.loadtxt(path, delimiter=',', usecols=0)# nanosecond, all files have the same time axis
        self.p_arr = time_to_momentum_axis(self.time_arr*1e-9, L=self.L, t0=self.t0) # magnitude of momentum in a.u.
        # correct the time_arr
        self.time_arr = self.time_arr - self.t0*1e9 # nanosecond

    def load_laser_shots(self, fn=None, dir_path=None): # Load the number of laser shots
        if fn is None:
            fn = f"{self.TARGET}_averaged_count_TDC2228A.dat"
        if dir_path is None:
            dir_path = self.data_dir_path
        path = os.path.join(dir_path, fn)
        self.detected_angles2, self.avg_count, self.tot_laser_shots = np.loadtxt(path).T

    def load_theory(self, E0f=None, target=None, fn=None, dir_path=None, skiprows=10): # Load the theory
        if target is None:
            target = self.TARGET
        if E0f is None:
            E0f = self.E0f
        if fn is None:
            fn = f"DCS_{target}_{int(E0f)}_ev.csv"
        if dir_path is None:
            dir_path = self.data_dir_path
        path = os.path.join(dir_path, fn)
        self.ang_theory, self.dcs_theory = np.loadtxt(path, delimiter=',', unpack=True, skiprows=skiprows)
        return self.ang_theory, self.dcs_theory #this is useful when having multiple theories

    # Calculations
    def correct_yield_to_same_num_of_laser_shots(self, unified_num_of_laser_shots=None):
        if unified_num_of_laser_shots is None:
            unified_num_of_laser_shots = np.mean(self.tot_laser_shots)
        self.unified_num_of_laser_shots = unified_num_of_laser_shots
        self.yield_tof_arr_2d = self.yield_tof_arr_2d * self.unified_num_of_laser_shots / self.tot_laser_shots[None, :]

    def calc_semiclassical_trajectory(self, delta=0.02, trajectroy_type='long'):
        self.delta = delta
        self.ke_scatter = self.E / self.Up
        self.ke_backscatter = search_traj_table("ke_backscatter", "ke_scatter", self.ke_scatter, trajectroy_type=trajectroy_type)
        self.born_phase = search_traj_table("born_phase", "ke_scatter", self.ke_scatter, trajectroy_type=trajectroy_type) # Not needed but good to have!
        self.return_phase = search_traj_table("return_phase", "ke_scatter", self.ke_scatter, trajectroy_type=trajectroy_type) # Not needed but good to have!
        # convert Up to au
        self.up_au = energy_ev_to_au(self.Up)
        self.ke_scatter = self.ke_scatter * self.up_au # convert from units of up to au
        self.ke_backscatter = self.ke_backscatter * self.up_au
        # Find limit of the uncertainty in energy provided as delta
        self.ke_scatter_min = self.ke_scatter * (1 - delta)
        self.ke_scatter_max = self.ke_scatter * (1 + delta)
        self.ke_backscatter_min = self.ke_backscatter * (1 - delta)
        self.ke_backscatter_max = self.ke_backscatter * (1 + delta)
        # From the ke_scatter and ke_backscatter, get the vector potential and momentum at rescattering
        self.ar = vector_potential_rescatter(self.ke_scatter, self.ke_backscatter)
        self.pr = momentum_rescatter_amp_ke(self.ke_scatter)
        self.ar_min = vector_potential_rescatter(self.ke_scatter_min, self.ke_backscatter_min)
        self.pr_min = momentum_rescatter_amp_ke(self.ke_scatter_min)
        self.ar_max = vector_potential_rescatter(self.ke_scatter_max, self.ke_backscatter_max)
        self.pr_max = momentum_rescatter_amp_ke(self.ke_scatter_max)

    def calc_dcs(self, apply_jacobian=True, interpolation=False, interpol_factor=1, jacobian_degree=3, interpol_kind='cubic', ):
        if apply_jacobian == True:
            self.dcs = self._calc_dcs_jacobian(interpolation=interpolation, interpol_factor=interpol_factor, jacobian_degree=jacobian_degree, interpol_kind=interpol_kind)
        else:
            self.dcs = self._calc_dcs_no_jacobian(interpolation=interpolation, interpol_factor=interpol_factor, interpol_kind=interpol_kind)

    def _calc_dcs_jacobian(self, interpolation=False, interpol_factor=10, jacobian_degree=2, interpol_kind='cubic'):
        dcs = np.zeros_like(self.detected_angles, dtype=np.float64) # differential cross section
        self.scatter_angle_deg_arr = np.zeros_like(self.detected_angles, dtype=np.float64)
        if interpolation:
            p_arr = np.linspace(self.p_arr[0], self.p_arr[-1], len(self.p_arr)*interpol_factor)
        else:
            p_arr = self.p_arr
        # Loop over the detected angle
        for i, det_angle in enumerate(self.detected_angles):
            # From the ar and pr, get the detected momentum at the detected angle
            # pd = detected_momentum_at_detected_angle(ar, pr, det_angle)
            scattering_ang_min = scattering_angle(det_angle, self.ar_min, self.pr_min) # degree
            scattering_ang_max = scattering_angle(det_angle, self.ar_max, self.pr_max)
            pd_min = detected_momentum_at_scattring_angle(self.ar_min, self.pr_min, scattering_ang_min)
            pd_max = detected_momentum_at_scattring_angle(self.ar_max, self.pr_max, scattering_ang_max)
            # print("Detected angle: ", det_angle)
            # print("Ke Range [Up] ", pd_min**2/(2*up_au), pd_max**2/(2*up_au))
            if pd_min < self.p_arr[0] or pd_max > self.p_arr[-1]:
                print("Warning: The edge of the momentum array is reached!")
                continue
            # Find the coressponding yield between pd1 and pd2 in the data
            indx_min = find_index_near(p_arr, pd_min)
            indx_max = find_index_near(p_arr, pd_max)
            spectrum = self.yield_tof_arr_2d[:, i]
            spectrum = spectrum[::-1] / (self.p_arr**jacobian_degree)
            if interpolation:
                spline = interp1d(self.p_arr, spectrum, kind=interpol_kind)
                spectrum = spline(p_arr)
            # Apply the jacobian
            dcs[i] = np.sum(spectrum[indx_min:indx_max+1]) # assuming that angle i corresponds to the same index in the yield_t
            self.scatter_angle_deg_arr[i] = (scattering_ang_max + scattering_ang_min) / 2  #scattering_angle(det_angle, self.ar, self.pr) # degree
        return dcs

    def _calc_dcs_no_jacobian(self, interpolation=False, interpol_factor=10, interpol_kind='cubic', ):
        dcs = np.zeros_like(self.detected_angles, dtype=np.float64) # differential cross section
        self.scatter_angle_deg_arr = np.zeros_like(self.detected_angles, dtype=np.float64)
        if interpolation:
            time_arr = np.linspace(self.time_arr[0], self.time_arr[-1], len(self.time_arr)*interpol_factor)
        else:
            time_arr = self.time_arr
        # Loop over the detected angle
        for i, det_angle in enumerate(self.detected_angles):
            # From the ar and pr, get the detected momentum at the detected angle
            # pd = detected_momentum_at_detected_angle(ar, pr, det_angle)
            scattering_ang_min = scattering_angle(det_angle, self.ar_min, self.pr_min) # degree
            scattering_ang_max = scattering_angle(det_angle, self.ar_max, self.pr_max)
            pd_min = detected_momentum_at_scattring_angle(self.ar_min, self.pr_min, scattering_ang_min)
            pd_max = detected_momentum_at_scattring_angle(self.ar_max, self.pr_max, scattering_ang_max)
            # print("Detected angle: ", det_angle)
            # print("Ke Range [Up] ", pd_min**2/(2*up_au), pd_max**2/(2*up_au))
            if pd_min < self.p_arr[0] or pd_max > self.p_arr[-1]:
                print("Warning: The edge of the momentum array is reached!")
                continue
            # Find the coressponding yield between pd1 and pd2 in the data
            # Note: the larger the momentum the smaller the tof
            tof_max = momentum_au_to_tof_ns(pd_min, t0_ns=0, L=self.L) # ns
            tof_min = momentum_au_to_tof_ns(pd_max, t0_ns=0, L=self.L) # ns
            # t0_ns=0 because we already corrected the time axis
            indx_min = find_index_near(time_arr, tof_min)
            indx_max = find_index_near(time_arr, tof_max)
            spectrum = self.yield_tof_arr_2d[:, i]
            if interpolation:
                spline = interp1d(self.time_arr, spectrum, kind=interpol_kind)
                spectrum = spline(time_arr)
            dcs[i] = np.sum(spectrum[indx_min:indx_max+1]) # assuming that angle i corresponds to the same index in the yield_t
            self.scatter_angle_deg_arr[i] = (scattering_ang_max + scattering_ang_min) / 2  #scattering_angle(det_angle, self.ar, self.pr) # degree
        return dcs

    def calculate_momentum_transfer(self, unit='A-1'):
        """ Unit are either 'A-1' or 'au'
        A-1 is Angstrom^-1
        """
        # Find momentum transfer
        # q = 2 p_r sin(phi/2)
        self.momentum_transfer = 2 * self.pr * np.sin(np.deg2rad(self.scatter_angle_deg_arr)/2) # au
        if unit == 'A-1':
            self.momentum_transfer = momentum_au_to_angstrom_inv(self.momentum_transfer)

    def calculate_error(self, use_current_dcs=False, normalize=True):
        # dcs should only be found from no_jacobian since the jacobian changes the real yield count
        if use_current_dcs: # for the sake of preformance when dcs is from no jacobian
            dcs = self.dcs
        else:
            dcs = self._calc_dcs_no_jacobian(interpolation=False)
            # dcs[dcs == 0] = min(dcs[dcs > 0]) * 1e-1 # to avoid -inf in DCS error, you can remove this line without affecting the results
        self.error = 1 / np.sqrt(dcs)
        if normalize:
            self.error /= np.mean(dcs[:4])


    # post calculations
    def normalize(self, normalization_type:str='avg'):
        if normalization_type == 'avg':
            norm_factor = np.mean(self.dcs[:4])
        elif normalization_type == 'max':
            norm_factor = max(self.dcs[:4])
        elif normalization_type == 'none':
            norm_factor = 1
        self.dcs /= norm_factor

    def calculate_dcs_diff(self, window=5):
        # 0 to e-10
        self.dcs[self.dcs <= 1e-5] = np.min(self.dcs[self.dcs > 0]) * 1e-2 # to avoid -inf in DCS, this can be removed withou affecting the results. The dft needs this adjustment to work
        self.dcs_log = np.log10(self.dcs)
        self.dcs_log_avg = np.convolve(self.dcs_log, np.ones(window)/window, mode='same')
        self.dcs_log_diff = self.dcs_log - self.dcs_log_avg
    # FFT
    def fft_momentum_transfer(self, dq=-0.02, auto_remove_dc_offset=True, calc_phase=False, power_spectrum=True, start_ind=0, zero_padding=0):
        # because the LIED scan is taken from angle -4 rather than 0, we need to remove mom_transf duplicate vlues
        # there for choose start_ind>0 untill you don't see the duplicate error (ValueError: Expect x to not have duplicates)
        # make equallly spaced momentum transfer and dcs using interpolation
        q_arr = np.arange(self.momentum_transfer[0+start_ind], self.momentum_transfer[-1], dq) # dq us usually negative
        # self.calculate_dcs_diff() # has to be run before runing this function
        dcs_interp = interp1d(self.momentum_transfer[start_ind:], self.dcs_log_diff[start_ind:], kind='cubic')
        dcs = dcs_interp(q_arr)
        # make -inf as e-10
        dcs[dcs == -np.inf] = -10
        self.freq, self.power_spec = fft(dcs, dq, auto_remove_dc_offset=auto_remove_dc_offset, calc_phase=calc_phase, power_spectrum=power_spectrum, zero_padding=zero_padding)

    def calc_moving_avg(self, y, window=3):# not good with smoothing, use calc_smoothed_line instead
        return np.convolve(y, np.ones(window)/window, mode='same')

    def calc_smoothed_line(self, x, y, f=5, kind='cubic'):
        spline = interp1d(x, y, kind=kind)
        x = np.linspace(x[0], x[-1], len(x)*f)
        return x, spline(x)

    # Plotting
    def creat_dcs_fig(self, title=None, E0f=None, tfontsize=12, y_scale="log",figsize=(5,4),\
                      xlabel=None, ylabel=None, fontsize=11, fontweight='bold'):
        # Labeling
        self.fig_dcs, self.ax_dcs = plt.subplots(tight_layout=True, figsize=figsize)
        if xlabel is None:
            xlabel='Scattering angle [deg]'
        if ylabel is None:
            ylabel="DCS [arb. unit]"#'Differential Cross-Section [arb. unit]'
        self.ax_dcs.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)
        self.ax_dcs.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)
        if E0f is None:
            E0f = self.E0f
        if title is None:
            target_label = self.TARGET.replace("_", " ")
            title = f"{target_label} at {int(E0f)}eV ±{round(self.delta*100)}% | {self.WL}um | {self.I0}TW/cm2 | Up = {round(self.Up)}eV"
        self.ax_dcs.set_title(title, fontsize=tfontsize, fontweight='bold')
        self.ax_dcs.set_yscale(y_scale)

    def creat_mom_trans_fig(self, title=None, E0f=None, tfontsize=12, y_scale="log",\
                        xlabel=None, ylabel=None, fontsize=11, fontweight='bold', figsize=(5,4)):
        self.fig_mt, self.ax_mt = plt.subplots(tight_layout=True, figsize=figsize)
        # Labeling
        if xlabel is None:
            xlabel='Momentum Transfer [$Å^{-1}$]'
        if ylabel is None:
            ylabel="DCS [arb. unit]"#'Differential Cross-Section [arb. unit]'
        self.ax_mt.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)
        self.ax_mt.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)
        if E0f is None:
            E0f = self.E0f
        if title is None:
            target_label = self.TARGET.replace("_", " ")
            title = f"{target_label} at {int(E0f)}eV ±{round(self.delta*100)}% | {self.WL}um | {self.I0}TW/cm2 | Up = {round(self.Up)}eV"
        self.ax_mt.set_title(title, fontsize=tfontsize, fontweight='bold')
        self.ax_mt.set_yscale(y_scale)

    def creat_fft_fig(self, title=None, tfontsize=12, y_scale="linear",figsize=(4,4),\
                        xlabel=None, ylabel=None, fontsize=11, fontweight='bold'):
        self.fig_fft, self.ax_fft = plt.subplots(tight_layout=True, figsize=figsize)
        # Labeling
        if xlabel is None:
            xlabel='Internuclear Separation [$Å$]'
        if ylabel is None:
            ylabel='Normalized Power Spectrum'
        self.ax_fft.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)
        self.ax_fft.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)
        if title is None:
            target_label = self.TARGET.replace("_", " ")
            title = f"DFT{target_label} at {int(self.E0f)}eV ±{round(self.delta*100)}% | {self.WL}um | {self.I0}TW/cm2 | Up = {round(self.Up)}eV"
        self.ax_fft.set_title(title, fontsize=tfontsize, fontweight='bold')
        self.ax_fft.set_yscale(y_scale)

    def set_xylim(self, xlim=None, ylim=None, ax=None):
        if ax is None:
            ax = self.ax_dcs
        if xlim is None:
            xlim = (min(self.scatter_angle_deg_arr)-5, 190)
        if ylim is None:
            ylim = (min(self.dcs)/2, 20)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    def plot_dcs(self,ax=None, marker='o', linestyle='', markersize=7, line_width=2, capsize=4, color='blue', label='LIED', errorbar=True,\
                 smooth_line=False, ma_linestyle='-', ma_line_width=2, majorticks=20, minorticks=10, start_ind=0, y_offset=1.):
        if color is None:
            color = self._get_color()
        if ax is None:
            ax = self.ax_dcs
        if smooth_line: # remove linestyle because it will be plotted with the moving average. So only plot the markers
            linestyle = ''
        if errorbar:
            ax.errorbar(self.scatter_angle_deg_arr[start_ind:], self.dcs[start_ind:]*y_offset, self.error[start_ind:]*y_offset, linestyle=linestyle, lw=line_width, marker=marker, capsize=capsize, color=color, markersize=markersize, label=label)
        else:
            ax.plot(self.scatter_angle_deg_arr[start_ind:], self.dcs[start_ind:]*y_offset, linestyle=linestyle, lw=line_width, marker=marker, color=color, markersize=markersize, label=label)
        if smooth_line:
            x, y = self.calc_smoothed_line(self.scatter_angle_deg_arr[start_ind:], np.log10(self.dcs[start_ind:]*y_offset))
            ax.plot(x, 10**y, color=color, linestyle=ma_linestyle, lw=ma_line_width)
        ax.xaxis.set_major_locator(MultipleLocator(majorticks))
        ax.xaxis.set_minor_locator(MultipleLocator(minorticks))

    def plot_theory(self, ax=None, E0f=None, target=None, fn=None, dir_path=None, skiprows=10,\
                    theory_y_offset=0, theory_label='CED', color='black', line_width=2,\
                    normalize=True):
        if ax is None:
            ax = self.ax_dcs
        #### Load CED (convential electron diffraction) and compare their diffraction crossection data (from NIST) with LIED ####
        if E0f is None:
            E0f = self.E0f
        ang_theory, dcs_theory = self.load_theory(E0f=E0f, target=target, fn=fn, dir_path=dir_path, skiprows=skiprows)
        if normalize:
            dcs_theory = dcs_theory / np.max(dcs_theory[-5:])
        dcs_theory += theory_y_offset
        ax.plot(ang_theory, dcs_theory, color=color, label=theory_label, lw=line_width)

    def plot_momentum_transfer(self, ax=None, marker='o', linestyle='', markersize=7, line_width=2, capsize=4, color='blue', label='LIED', errorbar=True,
                               smooth_line=False, ma_linestyle="-", ma_line_width=2, start_ind=0, major_ticks=0.5, minor_ticks=0.1, y_offset=1.):
        # start_ind is used to avoid the duplicate values in the momentum transfer since the LIED scan is typically taken from angle -4 rather than 0
        if color is None:
            color = self._get_color()
        if ax is None:
            ax = self.ax_mt
        if smooth_line: # remove linestyle because it will be plotted with the moving average. So only plot the markers
            linestyle = ''
        if errorbar:
            ax.errorbar(self.momentum_transfer[start_ind:], self.dcs[start_ind:]*y_offset, self.error[start_ind:]*y_offset, marker=marker, linestyle=linestyle, markersize=markersize, capsize=capsize, lw=line_width, color=color, label=label)
        else:
            ax.plot(self.momentum_transfer[start_ind:], self.dcs[start_ind:]*y_offset, marker=marker, linestyle=linestyle, markersize=markersize, lw=line_width, color=color, label=label)
        if smooth_line:
            x, y = self.calc_smoothed_line(self.momentum_transfer[start_ind:], np.log10(self.dcs[start_ind:]*y_offset)) # start from 2 (simlar to fft_start_ind) to avoid (ValueError: Expect x to not have duplicates)
            ax.plot(x, 10**y, color=color, linestyle=ma_linestyle, lw=ma_line_width)
        ax.xaxis.set_major_locator(MultipleLocator(major_ticks))
        ax.xaxis.set_minor_locator(MultipleLocator(minor_ticks))

    def plot_momentum_transfer_diff(self, ax=None, marker='o', linestyle='', markersize=7, line_width=2, capsize=4, color='blue', label='LIED', errorbar=True,
                                    smooth_line=False, ma_linestyle='-', ma_line_width=2, start_ind=0):
        if color is None:
            color = self._get_color()
        if ax is None:
            ax = self.ax_mt
        if smooth_line: # remove linestyle because it will be plotted with the moving average. So only plot the markers
            linestyle = ''
        if errorbar:
            ax.errorbar(self.momentum_transfer[start_ind:], 10**self.dcs_log_diff[start_ind:], self.error[start_ind:], marker=marker, linestyle=linestyle, markersize=markersize, capsize=capsize, lw=line_width, color=color, label=label)
        else:
            ax.plot(self.momentum_transfer[start_ind:], 10**self.dcs_log_diff[start_ind:], marker=marker, linestyle=linestyle, markersize=markersize, lw=line_width, color=color, label=label)
        if smooth_line:
            x, y = self.calc_smoothed_line(self.momentum_transfer[start_ind:], self.dcs_log_diff[start_ind:])
            ax.plot(x, 10**y, color=color, linestyle=ma_linestyle, lw=ma_line_width)

    def plot_mt_theory(self, ax=None, E0f=None, target=None, fn=None, dir_path=None, skiprows=10,\
                    theory_y_offset=0, theory_label='CED', color='black', line_width=2,\
                    normalize=True, unit='A-1'):
        """unit are either 'A-1' or 'au'"""
        if ax is None:
            ax = self.ax_mt
        #### Load CED (convential electron diffraction) and compare their diffraction crossection data (from NIST) with LIED ####
        if E0f is None:
            E0f = self.E0f
        ang_theory, dcs_theory = self.load_theory(E0f=E0f, target=target, fn=fn, dir_path=dir_path, skiprows=skiprows)
        mom_trans = 2 * self.pr * np.sin(np.deg2rad(ang_theory)/2)
        if unit == 'A-1':
            mom_trans = momentum_au_to_angstrom_inv(mom_trans)
        if normalize:
            dcs_theory = dcs_theory / np.max(dcs_theory[-5:])
        dcs_theory += theory_y_offset
        ax.plot(mom_trans, dcs_theory, color=color, label=theory_label, lw=line_width)

    def plot_fft(self, ax=None, marker='o', linestyle='', markersize=7, line_width=2, capsize=4, color='blue', label='LIED',
                 smooth_line=False, ma_linestyle='-', major_ticks=0.5, minor_ticks=0.1, normalize=False):
        if color is None:
            color = self._get_color()
        if ax is None:
            ax = self.ax_fft
        if smooth_line: # remove linestyle because it will be plotted with the moving average. So only plot the markers
            linestyle = ''
        # make fft positive
        freq = np.abs(self.freq[::-1])
        power_spec = self.power_spec[::-1]
        if normalize:
            power_spec = power_spec / np.max(power_spec)
        ax.plot(freq, power_spec, marker=marker, linestyle=linestyle, markersize=markersize, lw=line_width, color=color, label=label)
        if smooth_line:
            x, y = self.calc_smoothed_line(freq, power_spec)
            ax.plot(x, y, color=color, linestyle=ma_linestyle, lw=line_width)
        ax.xaxis.set_major_locator(MultipleLocator(major_ticks))
        ax.xaxis.set_minor_locator(MultipleLocator(minor_ticks))

    def show(self,):
        plt.tight_layout()
        plt.legend()
        plt.show()

    # Save
    def save_dcs_fig(self, E0f=None, save_dir_path=None, save_fn_path=None, dpi=300, target=None, fig=None, ax=None):
        if fig is None:
            fig = self.fig_dcs
        if ax is None:
            ax = self.ax_dcs
        if target is None:
            target = self.TARGET
        if E0f is None:
            E0f = self.E0f
        if save_fn_path is None:
            save_fn_path = f"DCS_{target}_{int(E0f)}eV_{self.WL}um_{self.I0}TW_Up{round(self.Up)}ev"
        if save_dir_path is None:
            save_dir_path = self.data_dir_path
        path = os.path.join(save_dir_path, save_fn_path)
        fig.tight_layout()
        ax.legend()
        fig.savefig(f"{path}.png", dpi=dpi)

    def save_momentum_transfer_fig(self, save_dir_path=None, save_fn_path=None, dpi=300, target=None, E0f=None, fig=None, ax=None):
        if fig is None:
            fig = self.fig_mt
        if ax is None:
            ax = self.ax_mt
        if target is None:
            target = self.TARGET
        if E0f is None:
            E0f = self.E0f
        if save_fn_path is None:
            save_fn_path = f"Momentum_Transfer_{target}_{int(E0f)}eV_{self.WL}um_{self.I0}TW_Up{round(self.Up)}ev"
        if save_dir_path is None:
            save_dir_path = self.data_dir_path
        path = os.path.join(save_dir_path, save_fn_path)
        fig.tight_layout()
        ax.legend()
        fig.savefig(f"{path}.png", dpi=dpi)

    def save_fft_fig(self, save_dir_path=None, save_fn_path=None, dpi=300, target=None, E0f=None, fig=None, ax=None, legend=False):
        if fig is None:
            fig = self.fig_fft
        if ax is None:
            ax = self.ax_fft
        if target is None:
            target = self.TARGET
        if E0f is None:
            E0f = self.E0f
        if save_fn_path is None:
            save_fn_path = f"FFT_{target}_{int(E0f)}eV_{self.WL}um_{self.I0}TW_Up{round(self.Up)}ev"
        if save_dir_path is None:
            save_dir_path = self.data_dir_path
        path = os.path.join(save_dir_path, save_fn_path)
        fig.tight_layout()
        if legend:
            ax.legend()
        fig.savefig(f"{path}.png", dpi=dpi)

    def save_calculations(self,save_dir_path=None, save_fn_path=None, dpi=300, target=None, E0f=None):
        if target is None:
            target = self.TARGET
        if E0f is None:
            E0f = self.E0f
        if save_fn_path is None:
            save_fn_path = f"LIED_Calculations_{target}_{int(E0f)}eV_{self.WL}um_{self.I0}TW_Up{round(self.Up)}ev"
        if save_dir_path is None:
            save_dir_path = self.data_dir_path
        path = os.path.join(save_dir_path, save_fn_path)
        with open(f"{path}.txt", 'a+') as f:
            f.write(f"\nTARGET: {target}\nEnergy = {self.E}eV\nUp = {self.Up}eV\ndelta = {self.delta}\nt0 = {self.t0}\nL = {self.L}\nE0f = {E0f}\nWL = {self.WL}\nI0 = {self.I0}\n")
            f.write(f"KE Scattering [Up] = {self.ke_scatter/self.up_au}\n")
            f.write(f"KE BackScattering [Up] = {self.ke_backscatter/self.up_au}\n")
            f.write(f"Pr [au] = {self.pr}\n")
            f.write(f"Ar [au] = {self.ar}\n")
            f.write(f"Uncertainty [%] = {round(self.delta*100,1)}\n")
            f.write(f"Born Phase [rad] = {self.born_phase}\n")
            f.write(f"Number of laser shots = {self.unified_num_of_laser_shots}\n")


    def run(self, delta=0.05, apply_jacobian=True, save=False, show=True, has_theory=False, plot_fft=0,\
            show_mt=False, save_mt=False, save_calculations=False, unified_num_of_laser_shots=None,\
            title=None, errorbar=True, xlim=None, ylim=None, ax=None, ax_mt=None, ax_fft=None, xlim_mt=None, ylim_mt=None,\
            marker='o', linestyle='', markersize=7, capsize=4, label='LIED', plot=True, plot_mt=False,\
            color=None,tfontsize=12, y_scale="log", xlabel=None, ylabel=None, xlabel_mt=None, ylabel_mt=None,\
            jacobian_degree=3, interpolation=True, interpol_factor=1, interpol_kind='linear', theory_y_offset=0,\
            data_fn=None, dir_path=None, time_arr_fn=None, laser_shots_fn=None, E0f=None, target=None,\
            theory_fn=None, theory_dir_path=None, save_dir_path=None, save_fn_path=None, normalization_type='avg',\
            theory_label='CED', theory_color='black', line_width=2, normalize=True, mt_unit='A-1', median_threshold:int=None,
            show_fft=False, save_fft=False, fft_dq=-0.1, fft_auto_remove_dc_offset=True, fft_calc_phase=False,\
            fft_power_spectrum=True, fft_start_ind=0, zero_padding=0, window_diff=5, smooth_dcs=False, smooth_mt=False,\
            smooth_fft=False, smooth_mt_diff=False, plot_mt_diff=False, xlim_fft=None, ylim_fft=None, color_mt_diff=None,
            figsize=(5,4), figsize_fft=(4,4), normalize_fft=False, dcs_offset=1, mt_offset=1):

        # Load data
        self.load_data(fn=data_fn, dir_path=dir_path, median_threshold=median_threshold)
        self.load_time_arr(fn=time_arr_fn, dir_path=dir_path)
        self.load_laser_shots(fn=laser_shots_fn, dir_path=dir_path)
        # Load theory
        if has_theory:
            self.load_theory(E0f=E0f, target=target, fn=theory_fn, dir_path=theory_dir_path)
        # prepare
        self.correct_yield_to_same_num_of_laser_shots(unified_num_of_laser_shots=unified_num_of_laser_shots)
        # calculation
        self.calc_semiclassical_trajectory(delta=delta)
        self.calc_dcs(apply_jacobian=apply_jacobian,  jacobian_degree=jacobian_degree, interpolation=interpolation, interpol_factor=interpol_factor, interpol_kind=interpol_kind)
        if plot_mt or show_mt or save_mt:
            self.calculate_momentum_transfer(unit=mt_unit)
        if errorbar:
            self.calculate_error(use_current_dcs=False, normalize=normalize)
        # Post calculation
        if normalize:
            self.normalize(normalization_type=normalization_type)
        # FFT
        self.calculate_dcs_diff(window=window_diff) # used for fft and can be used with other plots as well such as dcs and mom_trans
        if plot_fft or show_fft or save_fft:
            self.fft_momentum_transfer(dq=fft_dq, auto_remove_dc_offset=fft_auto_remove_dc_offset, calc_phase=fft_calc_phase, power_spectrum=fft_power_spectrum, start_ind=fft_start_ind, zero_padding=zero_padding)
        # plotting
        if show or save or plot:
            if ax is None:
                self.creat_dcs_fig(title=title, E0f=E0f, tfontsize=tfontsize, y_scale=y_scale,figsize=figsize,\
                                    xlabel=xlabel, ylabel=ylabel, fontsize=11, fontweight='bold')
                self.set_xylim(xlim=xlim, ylim=ylim, ax=ax)
            self.plot_dcs(ax=ax, marker=marker, linestyle=linestyle, markersize=markersize, line_width=line_width, capsize=capsize, color=color, label=label, errorbar=errorbar,
                          smooth_line=smooth_dcs, ma_linestyle=linestyle, ma_line_width=line_width, start_ind=fft_start_ind, y_offset=dcs_offset)
            if has_theory:
                self.plot_theory(ax=ax, E0f=E0f, target=target, fn=theory_fn, dir_path=theory_dir_path, skiprows=10,\
                                theory_y_offset=theory_y_offset, theory_label=theory_label, color=theory_color, line_width=line_width, normalize=normalize)
            if save:
                self.save_dcs_fig(E0f=E0f, save_dir_path=save_dir_path, save_fn_path=save_fn_path, dpi=300, target=target)
        if show_mt or save_mt or plot_mt:
            if ax_mt is None:
                self.creat_mom_trans_fig(title=title, E0f=E0f, tfontsize=tfontsize, y_scale=y_scale,figsize=figsize,\
                                    xlabel=xlabel_mt, ylabel=ylabel_mt, fontsize=11, fontweight='bold')
                self.set_xylim(xlim=xlim_mt, ylim=ylim_mt, ax=self.ax_mt)
            self.plot_momentum_transfer(ax=ax_mt, marker=marker, linestyle=linestyle, markersize=markersize, line_width=line_width, capsize=capsize, color=color, label=label, errorbar=errorbar,
                                        smooth_line=smooth_mt, ma_linestyle=linestyle, ma_line_width=line_width, start_ind=fft_start_ind, y_offset=mt_offset)
            if plot_mt_diff:
                self.plot_momentum_transfer_diff(ax=ax_mt, marker=marker, linestyle=linestyle, markersize=markersize, line_width=line_width, capsize=capsize, color=color_mt_diff, label=label, errorbar=errorbar,
                                        smooth_line=smooth_mt_diff, ma_linestyle=linestyle, ma_line_width=line_width, start_ind=fft_start_ind)
            if has_theory:
                self.plot_mt_theory(ax=ax_mt, E0f=E0f, target=target, fn=theory_fn, dir_path=theory_dir_path, skiprows=10,\
                                theory_y_offset=theory_y_offset, theory_label=theory_label, color=theory_color, line_width=line_width, normalize=normalize)
            if save_mt:
                self.save_momentum_transfer_fig(save_dir_path=save_dir_path, save_fn_path=save_fn_path, dpi=300, target=target, E0f=E0f)
        if plot_fft or show_fft or save_fft:
            if ax_fft is None:
                self.creat_fft_fig(title="DFT of MT for "+title, tfontsize=tfontsize, y_scale='linear',figsize=figsize_fft,\
                                    xlabel='Distance [$Å$]', ylabel='Amplitude [arb. unit]', fontsize=11, fontweight='bold')
                self.set_xylim(xlim=xlim_fft, ylim=ylim_fft, ax=self.ax_fft)
            self.plot_fft(ax=ax_fft, marker=marker, linestyle=linestyle, markersize=markersize, line_width=line_width, capsize=capsize, color=color, label=label,
                          smooth_line=smooth_fft, ma_linestyle=linestyle, normalize=normalize_fft)
            if save_fft:
                self.save_fft_fig(save_dir_path=save_dir_path, save_fn_path=save_fn_path, dpi=300, target=target, E0f=E0f)
        if show:
            self.show()
        if save_calculations:
            self.save_calculations(save_dir_path=save_dir_path, save_fn_path=save_fn_path, dpi=300, target=target, E0f=E0f)

    @classmethod
    def _gen_id(cls):
        cls.N_OF_OBJS += 1
        return cls.N_OF_OBJS

    def _get_color(self, ):
        color = self.COLORS[self.id % len(self.COLORS)]
        if self.tab_color:
            color = "tab:" + color
        return color


def LIED_scan_energy(E_list, Up, t0, L, WL, I0, TARGET, E0f, data_dir_path, delta=0.05, apply_jacobian=True,\
            save=False, show=True,plot_mt=False, has_theory=False, plot=True,\
            show_mt=False, save_mt=False, save_calculations=False, unified_num_of_laser_shots=None,\
            title=None, errorbar=True, xlim=None, ylim=None, ax=None, ax_mt=None, xlim_mt=None, ylim_mt=None,\
            marker='o', linestyle='', markersize=7, capsize=4, label='LIED',\
            color="blue",tfontsize=12, y_scale="log", xlabel='Scattering angle [deg]', ylabel='Differential Cross-Section [arb. unit]',\
            jacobian_degree=3, interpolation=True, interpol_factor=1, interpol_kind='linear', theory_y_offset=0,\
            data_fn=None, dir_path=None, time_arr_fn=None, laser_shots_fn=None, target=None,\
            theory_fn=None, theory_dir_path=None, save_dir_path=None, save_fn_path=None, normalization_type='avg',\
            theory_label='CED', theory_color='black', line_width=2, normalize=True, mt_unit='A-1', median_threshold:int=None):
    """LIED scan energy
    """
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    plot_mt = plot_mt or show_mt or save_mt
    lied_obj_list = []
    for i, E in enumerate(E_list):
        color = colors[i%len(colors)]
        label = f"LIED {int(E)}eV"
        lied = LIED(E=E, Up=Up, t0=t0, L=L, WL=WL, I0=I0, TARGET=TARGET, E0f=E0f, data_dir_path=data_dir_path)
        lied.run(delta=delta, apply_jacobian=apply_jacobian, save=0, show=0, has_theory=has_theory, median_threshold=median_threshold,\
            show_mt=0, save_mt=0, plot=plot, plot_mt=plot_mt, save_calculations=save_calculations, unified_num_of_laser_shots=unified_num_of_laser_shots,\
            title=title, errorbar=errorbar, xlim=xlim, ylim=ylim, ax=ax, ax_mt=ax_mt, xlim_mt=xlim_mt, ylim_mt=ylim_mt,\
            marker=marker, linestyle=linestyle, markersize=markersize, capsize=capsize, label=label,\
            color=color,tfontsize=tfontsize, y_scale=y_scale, xlabel=xlabel, ylabel=ylabel,\
            jacobian_degree=jacobian_degree, interpolation=interpolation, interpol_factor=interpol_factor, interpol_kind=interpol_kind, theory_y_offset=theory_y_offset,\
            data_fn=data_fn, dir_path=dir_path, time_arr_fn=time_arr_fn, laser_shots_fn=laser_shots_fn, target=target,\
            theory_fn=theory_fn, theory_dir_path=theory_dir_path, save_dir_path=save_dir_path, save_fn_path=save_fn_path, normalization_type=normalization_type,\
            theory_label=f"{theory_label} {label}", theory_color=color, line_width=line_width, normalize=normalize, mt_unit=mt_unit)
        if ax is None and (plot or show or save):
            ax = lied.ax_dcs
            fig = lied.fig_dcs
        if ax_mt is None and (plot_mt or show_mt or save_mt):
            ax_mt = lied.ax_mt
            fig_mt = lied.fig_mt
        lied_obj_list.append(lied)
    if show or show_mt:
        lied.show()
    if save:
        if save_fn_path is None:
            save_fn_path = f"LIED_{TARGET}_{int(E0f)}eV_{WL}um_{I0}TW_Up{round(Up)}ev"

        lied.save_dcs_fig(E0f=E0f, save_dir_path=save_dir_path, save_fn_path=save_fn_path, dpi=300, target=target, fig=fig, ax=ax)
    if save_mt:
        if save_fn_path is None:
            save_fn_path = f"Momentum_Transfer_{TARGET}_{int(E0f)}eV_{WL}um_{I0}TW_Up{round(Up)}ev"
        else:
            save_fn_path = f'Momentum_Transfer_{save_fn_path}'
        lied.save_momentum_transfer_fig(save_dir_path=save_dir_path, save_fn_path=save_fn_path, dpi=300, target=target, E0f=E0f, fig=fig_mt, ax=ax_mt)
    return lied_obj_list




########################## Draft ##########################
#### Example of using the LIED class ####
    ### Typical Run with Xe
    # lied_100ev = LIED(
    # E = 100, # eV
    # Up = 61.1, # eV
    # t0 = -18.5E-9,
    # L = 0.54,
    # WL = 3.5, # um
    # I0 = 54, #TW/cm2
    # TARGET = 'Xe',

    # E0f = 100, #E
    # data_dir_path=r'.',
    # )
    # lied_100ev.load_data(fn=None, dir_path=None)
    # lied_100ev.load_time_arr(fn=None, dir_path=None)
    # lied_100ev.load_laser_shots(fn=None, dir_path=None)
    # lied_100ev.load_theory(E0f=None, target=None, fn=None, dir_path=None, skiprows=10)
    # # prepare
    # lied_100ev.correct_yield_to_same_num_of_laser_shots(unified_num_of_laser_shots=None)
    # # calculation
    # lied_100ev.calc_semiclassical_trajectory(delta=0.05)
    # lied_100ev.calc_dcs(apply_jacobian=True,  jacobian_degree=3, interpolation=True, interpol_factor=1, interpol_kind='linear')
    # lied_100ev.calculate_momentum_transfer()
    # lied_100ev.calculate_error(use_current_dcs=False, normalize=True)
    # # Post calculation
    # lied_100ev.normalize(normalization_type='avg') # 'max' , 'none' or 'avg' to normalize dcs and error
    # # plotting
    # lied_100ev.creat_dcs_fig(title=None, E0f=None, tfontsize=12, y_scale="log",\
    #                     xlabel='Scattering angle [deg]', ylabel='Differential Cross-Section [arb. unit]', fontsize=11, fontweight='bold')
    # lied_100ev.creat_mom_trans_fig(title=None, E0f=None, tfontsize=12, y_scale="log",\
    #                     xlabel='Momentum Transfer [au]', ylabel='Differential Cross-Section [arb. unit]', fontsize=11, fontweight='bold')
    # lied_100ev.set_xylim(xlim=None, ylim=(10**-3/2, 5), ax=None)
    # lied_100ev.set_xylim(xlim=(1, 5.5), ylim=(10**-3/2, 5), ax=lied_100ev.ax_mt)
    # lied_100ev.plot_dcs(ax=None, marker='o', linestyle='', markersize=7, line_width=2, capsize=4, color='blue', label='LIED',errorbar=True)
    # lied_100ev.plot_theory(ax=None, E0f=None, target=None, fn=None, dir_path=None, skiprows=10,\
    #                 theory_y_offset=0, theory_label='CED', color='black', line_width=2,normalize=True)
    # lied_100ev.plot_momentum_transfer(ax=None, marker='o', linestyle='', markersize=7, line_width=2, capsize=4, color='blue', label='LIED',)
    # lied_100ev.plot_mt_theory(ax=None, E0f=None, target=None, fn=None, dir_path=None, skiprows=10,\
    #                 theory_y_offset=0, theory_label='CED', color='black', line_width=2,normalize=True)
    # lied_100ev.show()
    # lied_100ev.save_dcs_fig(E0f=None, save_dir_path=None, save_fn_path=None, dpi=300, target=None)
    # lied_100ev.save_momentum_transfer_fig(save_dir_path=None, save_fn_path=None, dpi=300, target=None, E0f=None)
    # lied_100ev.save_calculations(save_dir_path=None, save_fn_path=None, dpi=300, target=None, E0f=None)