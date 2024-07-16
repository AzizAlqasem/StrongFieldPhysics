"""About Photoelectron Yield"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import time

from StrongFieldPhysics.time_of_flight.tof_to_energy import t2E, t2E_fixed_bins
from StrongFieldPhysics.time_of_flight.tof_to_momentum import t2P
from StrongFieldPhysics.calculations.calculations import up, gamma, photon_energy, channel_closure
from StrongFieldPhysics.parser.data_files import read_header, remove_empty_lines
from StrongFieldPhysics.calculations.fourier import fft, filter_signal, correct_phase

# 30 AZIZ
# 3 LOLO
# 29 GDO

class PE_yield:

    def __init__(self, file_name:str, label="", I=None, Up=None, wl=None, t0=-18.5, L=0.54,\
                 dir_path=None, wl_unit='nm', Ip=None, E_max=100, dE=0.05):
        """Convert TOF spectra to Energy

        Args:
            file_name (str): path to file
            label (str, optional): label for the plot. Defaults to "".
            I (float, optional): Intensity in TW/cm2. Defaults to None.
            Up (float, optional): Up in eV. Defaults to None.
            wl (float, optional): Wavelength in nm. Defaults to None.
            t0 (float, optional): t0 in ns. Defaults to -18.5.
            L (float, optional): Length in m. Defaults to 0.54.
        """
        self.label = label
        self.file_name = file_name
        self.dir_path = dir_path if dir_path else r'.'
        self.t0 = t0
        self.L = L
        self.I = I if I else np.nan
        if wl is not None:
            self.photon_energy = 1239.84193 / wl
        else:
            self.photon_energy = np.nan
        self.wl = wl if wl else np.nan
        if Up is None and I and wl:
            Up = up(I, wl/1000)
        self.Up = Up if Up else np.nan
        self.Ip = Ip if Ip else np.nan
        if Ip and Up:
            self.gamma = gamma(Up, Ip)
            self.channel_closure = channel_closure(Up, Ip, wl)
        else:
            self.gamma = np.nan
            self.channel_closure = np.nan

        self.wl_unit = wl_unit
        self.E_max = E_max
        self.dE = dE
        self._set_flags()

    def load_data(self, file_name=None, dir_path=None, num_of_bins=None):
        """Load data from file
        num_of_bins: for TDC2228A#151ps, it should be 2038 (2048 - 10)
                    where 10 is dead bins that you can verify by looking at raw data!
        """
        if file_name:
            self.file_name = file_name
        if dir_path:
            self.dir_path = dir_path
        self.file_path = os.path.join(self.dir_path, self.file_name)
        data = np.loadtxt(self.file_path, delimiter=",")
        self.tof = data[:num_of_bins, 0] # ns, Just use the same time array in the data file. Which is used to find T0.
        self.count = data[:num_of_bins, 1]

    def correct_tof(self, t0=None):
        """Correct time of flight by subtracting t0"""
        t0 = t0 if t0 else self.t0
        if self.is_tof_corrected is False:
            self.tof = self.tof - t0
            self.is_tof_corrected = True

    def convert_tof_to_energy(self, normalize=True, method='jacobian', E_max=None, dE=None, normalize_method='max'):
        """Convert TOF to Energy
        method: 'jacobian' or 'integration'
        """
        E_max = E_max if E_max else self.E_max
        dE = dE if dE else self.dE
        if self.is_tof_corrected is False:
            self.correct_tof()
        if method == 'jacobian':
            self.energy, self.yeild_E = t2E(self.tof*1e-9, self.count, L=self.L, t0=0)
        elif method == 'integration':
            self.energy, self.yeild_E = t2E_fixed_bins(self.tof*1e-9, self.count, dE_bin=dE, E_max=E_max, L=self.L, t0=0)
        else:
            raise ValueError(f"Unknown method: {method}")
        if normalize:
            if normalize_method == 'max':
                self.yeild_E = self.yeild_E / np.max(self.yeild_E)
            elif normalize_method == 'mean':
                self.yeild_E = self.yeild_E / np.mean(self.yeild_E)
        return self.energy, self.yeild_E

    def convert_tof_to_momentum(self):
        """Convert TOF to Momentum
        """
        if self.is_tof_corrected is False:
            self.correct_tof()
        self.momentum, self.yeild_P = t2P(self.tof*1e-9, self.count, L=self.L, t0=0)
        return self.momentum, self.yeild_P

    ### Fourier Transform ###
    def fourier_transform(self, x_type='energy', dt=None, dc_offset=0, auto_remove_dc_offset=True):
        """Fourier transform the signal"""
        if x_type == 'energy':
            # Energy has to be fixed bin
            dt = self.dE if dt is None else dt
            self.freq, self.power_spec, self.phase = fft(self.yeild_E, dt=self.dE, dc_offset=dc_offset, auto_remove_dc_offset=auto_remove_dc_offset, calc_phase=True)
        elif x_type == 'tof':
            dt = (self.tof[1] - self.tof[0]) if dt is None else dt
            self.freq, self.power_spec, self.phase = fft(self.count, dt, calc_phase=True)
        # return self.freq, self.power_spec

    def filter_noise(self, threshold, remove_high_freq_above=None, remove_low_freq_below=None):
        self.yeild_E = filter_signal(self.yeild_E, self.dE, threshold=threshold, remove_high_freq_above=remove_high_freq_above, remove_low_freq_below=remove_low_freq_below)

    def get_phase_at_freq(self, freq):
        """Get phase at a given frequency"""
        idx = np.argmin(np.abs(self.freq - freq))
        return self.phase[idx]

    # A method to get T0 and L0 using FFT.
    ## Summary: Max. amplitude at E=PE should be at the optimum T0 and L0.
    ## plot ampl(at PE) vs T0 and L0 (2d plot) to find the optimum values.
    def get_t0_L0(self, t0_range=(-20, 20), L_range=(0.50, 0.55), nop_t0=10, nop_L=10, eng_lim=(3, 10), min_energy=8,\
                lim_t0l_func=lambda t0, l : True, peak=1, up_unit=True):
        """Get T0 and L0 using FFT
        emb_lim: energy limit of the FFT in Up
        min_energy: minimum energy that a TDC can detect (in eV) for TDC2228A#151ps, it is about 8 eV
        lim_t0l_func: function to limit the T0 and L0 range, it is useful to spped up the calculation since the range
        of interest is usually a narrow line with a slop and offset (eg: T0(L) = -116.471L + 48.778) so you guess a function
        that return true when the T0 and L are between two lines centered around the optimum line.
        peak: 1 for the first peak (at one photon spacing), 2 for the second peak, etc. of the fourier transform
        """
        t0s = np.linspace(*t0_range, nop_t0)
        Ls = np.linspace(*L_range, nop_L)
        ampl_2d = np.zeros((nop_t0, nop_L))
        phase_2d = np.zeros((nop_t0, nop_L))
        # check min_energy
        if up_unit:
            e1 = eng_lim[0] * self.Up
            e2 = eng_lim[1] * self.Up
        else:
            e1 = eng_lim[0]
            e2 = eng_lim[1]
        assert(e1 > min_energy, f"Minimum energy that a TDC can detect is {min_energy} eV")
        # Energy bound has to start at n photon energy such that the phase make sense.
        # Thuse we slightly shift it to the nearest n Photon energy.
        E1 = np.round(e1/ self.photon_energy) * self.photon_energy
        E2 = np.round(e2/ self.photon_energy) * self.photon_energy
        freq0 = 1 / self.photon_energy * peak
        for i, t0 in enumerate(t0s):
            for j, L in enumerate(Ls):
                if lim_t0l_func(t0, L) == False:
                    ampl_2d[i, j] = np.nan
                    phase_2d[i, j] = np.nan
                    continue
                energy, yield_E = t2E_fixed_bins(self.tof*1e-9 - t0*1e-9, self.count, dE_bin=self.dE, E_max=self.E_max, L=L, t0=0)
                yld_log_diff = self.calculate_diff_yield(yield_E)
                idx1 = np.argmin(np.abs(energy - E1))
                idx2 = np.argmin(np.abs(energy - E2))
                energy = energy[idx1:idx2]
                yld_log_diff = yld_log_diff[idx1:idx2]
                freq, power, phase = fft(yld_log_diff, dt=self.dE, dc_offset=0, auto_remove_dc_offset=True, calc_phase=True)
                idx = np.argmin(np.abs(freq - freq0))
                ampl_2d[i, j] = power[idx]
                phase_2d[i, j] = correct_phase(phase[idx]) / (2*np.pi) # 0 to 1
        return t0s, Ls, ampl_2d, phase_2d

    def calculate_diff_yield(self, yield_E, window_n_photons=1):
        n_of_bins_per_photon = self.photon_energy / self.dE
        window = int(n_of_bins_per_photon * window_n_photons)
        yld_log = np.log10(yield_E)
        yld_log_avg = np.convolve(yld_log, np.ones(window)/window, mode='same')
        return yld_log - yld_log_avg

    ### Plotting ###

    def creat_figure(self, figsize=None, dpi=None, tight_layout=True):
        """Create a figure"""
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi, tight_layout=tight_layout)


    def plot(self, ax=None, label=None, xlim=None, ylim=None, sep_yield=0, lw=2,\
             linestyle='-', color=None, x_type='energy', yscale=None, major_minor_ticks:list=None,\
            show_I=True, show_Up=True, show_wl=False, show_label=True, show_gamma=True,\
                show_channel_closure=False, show_photon_energy=False, offset_yield=0, auto_ylim=False,\
                normalize=False, skip_data_step:int=0, plot_type='plot', ls2='x'):
        """Plot the yield vs energy, momentum, or time"""
        ax = ax if ax else self.ax

        if x_type == 'energy':
            x = self.energy
            y = self.yeild_E
        elif x_type == 'energy_up':
            x = self.energy/self.Up
            y = self.yeild_E
        elif x_type == 'energy_ev_up': # dual x axis
            x = self.energy
            x2 = self.energy/self.Up
            y = self.yeild_E
        elif x_type == 'momentum':
            x = self.momentum
            y = self.yeild_P
        elif x_type == 'tof':
            x = self.tof
            y = self.count
        elif x_type == 'fft':
            x = self.freq
            y = self.power_spec
        elif x_type.lower() == 'fft-e': # the x axis has energy spacing
            x = 1 / self.freq
            y = self.power_spec
        else:
            raise ValueError(f"Unknown x_type: {x_type}")

        if normalize: # it might be already normalized
            y = y / np.max(y)

        if skip_data_step > 0:
            x = x[::skip_data_step]
            y = y[::skip_data_step]

        if sep_yield != 0:
            y = y * 10**(sep_yield)
        if offset_yield != 0:
            y = y + offset_yield

        label = self.gen_label(label=label, show_I=show_I, show_Up=show_Up, show_wl=show_wl,\
                               show_label=show_label, show_gamma=show_gamma, show_channel_closure\
                                =show_channel_closure, show_photon_energy=show_photon_energy)
        if plot_type.startswith('plot'):
            ax.plot(x, y, label=label, lw=lw, linestyle=linestyle, color=color)
        elif plot_type.startswith('stem'):
            ax.stem(x, y, label=label, markerfmt=' ', basefmt=' ')
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        if 'phase' in plot_type:
            # plot phase
            ax2 = ax.twinx()
            ax2.plot(x, self.phase, label='phase', lw=lw, marker='o', linestyle='', color='k', alpha=0.5)
            ax2.set_ylabel('Phase [rad]', fontsize=10)
            ax2.set_ylim([-np.pi, np.pi])
        if x_type == 'energy_ev_up':
            ax2 = ax.twiny()
            ax2.plot(x2, y, label=label, lw=lw, linestyle=ls2)
            ax2.set_xlabel('Photoelectron Energy [Up]', fontsize=10, fontweight='bold')
            ax2.xaxis.set_major_locator(MultipleLocator(1))
            ax2.xaxis.set_minor_locator(MultipleLocator(0.2))
            if xlim:
                ax2.set_xlim([xlim[0]/self.Up, xlim[1]/self.Up])
        if xlim:
            ax.set_xlim(xlim)
        if auto_ylim and xlim:
            # auto ylim ONLY works when xlim is set
            idx = np.where((x > xlim[0]) & (x < xlim[1]))
            ymax = np.max(y[idx]) * 1.01
            size = len(y[idx])
            ymin = np.min(y[idx][size//2:]) * 0.95 # ignore the first half when calculating ymin
            ylim = [ymin, ymax]
        if ylim:
            ax.set_ylim(ylim)
        if yscale:
            ax.set_yscale(yscale)
        if major_minor_ticks:
            major, minor = major_minor_ticks
            ax.xaxis.set_major_locator(MultipleLocator(major))
            ax.xaxis.set_minor_locator(MultipleLocator(minor))

    def plot_vline(self, ax=None, x=None, color='k', linestyle='--', lw=1):
        """Plot vertical lines"""
        ax = ax if ax else self.ax
        x = x if x else self.Up
        ax.axvline(x, color=color, linestyle=linestyle, lw=lw)

    def plot_atis_peaks(self, ax=None, fp=0, wl=None, n=10, color='k', linestyle='--', lw=1, alpha=1.0, up_unit=False):
        """Plot ATIS peaks"""
        ax = ax if ax else self.ax
        wl = wl if wl else self.wl
        pe = 1239.84193 / wl
        if up_unit:
            pe = pe / self.Up
        x = fp + np.arange(n) * pe
        for x_ in x:
            ax.axvline(x_, color=color, linestyle=linestyle, lw=lw, alpha=alpha)

    def plot_calibration(self, ax=None, up_m:list=[2, 4, 6, 8, 10], color='k', linestyle='--', lw=0.5):
        """Plot calibration as vertical lines"""
        ax = ax if ax else self.ax
        up = self.Up * np.array(up_m)
        label = ", ".join([str(up_) for up_ in up]) + " Up"
        ax.vlines(up, 0, 1, linestyles=linestyle, color=color, lw=lw, label=label)

    def label_plot(self, ax=None, xlabel=None, ylabel=None, title='', x_unit='eV', fontsize=10,\
                   tfontsize=12, fontweight='bold', show_I=False, show_Up=False, fig=None,\
                    show_wl=True, show_photon_energy=False, show_gamma=False, show_Ip=False, fig_title=False):
        """Label the plot"""
        ax = ax if ax else self.ax
        fig = fig if fig else self.fig
        xlabel = xlabel if xlabel else f"Photoelectron Energy [{x_unit}]"
        ylabel = ylabel if ylabel else "Relative Yield [a.u.]"
        title = self.gen_label(label=title, show_I=show_I, show_Up=show_Up, show_wl=show_wl,show_label=True,\
                               show_photon_energy=show_photon_energy, show_gamma=show_gamma, show_Ip=show_Ip)
        if fig_title:
            fig.suptitle(title, fontsize=tfontsize, fontweight='bold')
        else: # axis title
            ax.set_title(title, fontsize=tfontsize, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)

    def gen_label(self, label=None, show_I=True, show_Up=True, show_wl=False,\
                  show_label=True, show_t0=False, show_L=False, wl_unit=None,\
                  show_gamma=False, show_Ip=False, show_channel_closure=False, show_photon_energy=False):
        """Generate label from parameters"""
        label = label if label else self.label
        wl_unit = wl_unit if wl_unit else self.wl_unit
        labels = []
        if show_label:
            labels.append(str(label))
        if show_wl:
            if wl_unit == 'nm':
                labels.append(f"{round(self.wl,0)} {wl_unit}")
            elif wl_unit == 'um':
                labels.append(f"{round(self.wl/1000,2)} {wl_unit}")
            else:
                labels.append(f"{self.wl} {wl_unit}")
        if show_I:
            r = 0
            if self.I < 20:
                r = 1
            if self.I < 5:
                r = 2
            labels.append(f"{round(self.I,r)} $TW/cm^2$")
        if show_Up:
            r = 1
            if self.Up < 5:
                r = 2
            labels.append(f"Up={round(self.Up,r)} eV")
        if show_t0:
            labels.append(f"t0={self.t0} ns")
        if show_L:
            labels.append(f"L={self.L} m")
        if show_gamma:
            labels.append(fr"$\gamma$={round(self.gamma,1)}")
        if show_Ip:
            labels.append(f"Ip={round(self.Ip,2)} eV")
        if show_channel_closure:
            labels.append(f"n={round(self.channel_closure,1)}")
        if show_photon_energy:
            labels.append(f"PE={round(self.photon_energy,2)} eV")
        return " | ".join(labels)


    def save_plot(self, fig=None, file_name=None, dir_path=None, dpi=300):
        """Save the plot"""
        fig = fig if fig else self.fig
        if file_name is None:
            file_name = self.file_name
            # remove extension
            file_name = os.path.splitext(file_name)[0]
        dir_path = dir_path if dir_path else self.dir_path
        file_path = os.path.join(dir_path, file_name)
        self.fig.savefig(file_path, dpi=dpi)

    def _set_flags(self,):
        self.is_tof_corrected = False

    def save_data(self, dir_path=None, data_type='energy'):
        """Save data to file"""
        # prepare file name and directory
        file_name = self.file_name.replace('.txt', f'_{data_type}.txt')
        dir_path = dir_path if dir_path else os.path.join(self.dir_path, data_type+'_data')
        file_path = os.path.join(dir_path, file_name)
        os.makedirs(dir_path, exist_ok=True)  # create directory if it does not exist
        # load header
        header_list = read_header(self.file_path)[:-1]
        # add new header
        header_list.append(f'# Processed Data: {time.ctime()}')
        header_list.append(f'# t0 [ns]: {self.t0}')
        header_list.append(f'# L [m]: {self.L}')
        header_list.append(f'# Intensity [TW/cm2]: {self.I}')
        header_list.append(f'# Up [eV]: {self.Up}')
        header_list.append(f'# Ip [eV]: {self.Ip}')
        header_list.append(f'# Gamma: {self.gamma}')
        header_list.append(f'# Channel Closure: {self.channel_closure}')
        header_list.append(f'# Photon Energy [eV]: {self.photon_energy}')
        # data
        if data_type == 'energy':
            header_list.append(f"# Energy [eV], Yield [a.u.]")
            data = np.array([self.energy, self.yeild_E]).T
        elif data_type == 'momentum':
            header_list.append(f"# Momentum [a.u.], Yield [a.u.]")
            data = np.array([self.momentum, self.yeild_P]).T
        elif data_type == 'tof':
            header_list.append(f"# Time of Flight [ns], Counts [a.u.]")
            self.correct_tof() # make sure tof is corrected
            data = np.array([self.tof, self.count]).T
        # save data
        np.savetxt(file_path, data, delimiter=",", header="\n".join(header_list))

    ##### Export Data
    def export_data(self, file_name=None, dir_path=None, data_type='energy', add_calibration_header=True, normalize=False, method='jacobian', E_max=None, dE=None):
        """Export data to file"""
        # prepare file name and directory
        file_name = file_name if file_name else self.file_name.replace('.txt', f'_{data_type}.dat')
        dir_path = dir_path if dir_path else os.path.join(self.dir_path, data_type+'_data')
        file_path = os.path.join(dir_path, file_name)
        os.makedirs(dir_path, exist_ok=True)
        # load header
        # Exisiting header
        header_list = read_header(self.file_path)[:-1]
        header_list = remove_empty_lines(header_list, remove_hash=True)
        # calibration headers
        if add_calibration_header:
            header_list.append(f' -------- Time of flight converted to {data_type} using {method} ---------')
            header_list.append(f' Processed Date : {time.ctime()}')
            header_list.append(f' t0 [ns]: {self.t0}')
            header_list.append(f' L [m]: {self.L}')
            header_list.append(f' Intensity [TW/cm2]: {self.I}')
            header_list.append(f' Up [eV]: {self.Up}')
            header_list.append(f' Ip [eV]: {self.Ip}')
            header_list.append(f' Gamma: {self.gamma}')
            header_list.append(f' Channel Closure: {self.channel_closure}')
            header_list.append(f' Photon Energy [eV]: {self.photon_energy}')
            header_list.append(f' -----------------')
        # data
        self.correct_tof() # make sure tof is corrected
        if data_type == 'energy':
            header_list.append(f" Energy [eV], Yield [a.u.]")
            self.convert_tof_to_energy(normalize=normalize, method=method, E_max=E_max, dE=dE)
            data = np.array([self.energy, self.yeild_E]).T
        elif data_type == 'momentum':
            data = np.array([self.momentum, self.yeild_P]).T
        elif data_type == 'tof':
            data = np.array([self.tof, self.count]).T
        # save data
        np.savetxt(file_path, data, fmt='%1.6e', delimiter="\t", header="\n".join(header_list))

    ### Utility Functions ###
    def shift_energy_to_nereast_photon_energy(self, energy):
        """Shift energy to the nearest n photon energy"""
        return np.round(energy / self.photon_energy) * self.photon_energy




class PE_yield_collection(PE_yield):
    def __init__(self, file_names:list, labels:list=None, I:list=None, Up:list=None, wl:float=None, t0=-18.5, L=0.54,\
                 dir_path=None, Ip:float or list=None, wl_unit='nm'):
        """Convert TOF spectra to Energy

        Args:
            file_names (list): list of file names
            labels (list): list of labels
            I (list, optional): Intensity in TW/cm2. Defaults to None.
            Up (list, optional): Up in eV. Defaults to None.
            wl (float, optional): Wavelength in nm. Defaults to None.
            t0 (float, optional): t0 in ns. Defaults to -18.5.
            L (float, optional): Length in m. Defaults to 0.54.

        """
        self.file_names = file_names
        if labels is None:
            labels = [""] * len(file_names)
        self.labels = labels
        self.dir_path = dir_path if dir_path else r'.'
        if isinstance(t0, (list, np.ndarray)):
            self.t0 = t0
        else:
            self.t0 = [t0] * len(file_names)
        if isinstance(L, (list, np.ndarray)):
            self.L = L
        else:
            self.L = [L] * len(file_names)
        if isinstance(I, (list, np.ndarray)):
            self.I = I
        else:
            self.I = [I] * len(file_names)
        if isinstance(wl, (list, np.ndarray)):
            self.wl = wl
        else:
            self.wl = [wl] * len(file_names)
        if Up is None and isinstance(I, (list, np.ndarray)) and isinstance(self.wl, (list, np.ndarray)):
            Up = up(I, self.wl[0]/1000)
        self.Up = Up
        if isinstance(Ip, (list, np.ndarray)): # ionization potential
            self.Ip = Ip
        else:
            self.Ip = [Ip] * len(file_names)
        self.wl_unit = wl_unit

    def load_data(self, yileds_obj=None):
        if yileds_obj is None:
            self.yields = []
            # for file_name, label, I, Up, wl, ip in zip(self.file_names, self.labels, self.I, self.Up, self.wl, self.Ip):
                # yield_ = PE_yield(file_name, label=label, I=I, Up=Up, wl=wl,\
                #                 t0=self.t0, L=self.L, dir_path=self.dir_path, Ip=ip)
            for file_name, label, I, Up, wl, ip, t0, L in zip(self.file_names, self.labels, self.I, self.Up, self.wl, self.Ip, self.t0, self.L):
                yield_ = PE_yield(file_name, label=label, I=I, Up=Up, wl=wl,\
                                t0=t0, L=L, dir_path=self.dir_path, Ip=ip, wl_unit=self.wl_unit)
                yield_.load_data()
                self.yields.append(yield_)
        else:
            self.yields = yileds_obj

    def correct_tof(self, t0=None):
        """Correct time of flight by subtracting t0"""
        if t0 is None:
            t0 = self.t0
        elif not isinstance(t0, (list, np.ndarray)):
            t0 = [t0] * len(self.yields)
        for i, yield_ in enumerate(self.yields):
            yield_.correct_tof(t0=t0[i])

    def convert_tof_to_energy(self, normalize=True, method='jacobian', E_max=None, dE=None, normalize_method='max'):
        """Convert TOF to Energy"""
        for yield_ in self.yields:
            yield_.convert_tof_to_energy(normalize=normalize, method=method, E_max=E_max, dE=dE, normalize_method=normalize_method)

    def convert_tof_to_momentum(self):
        """Convert TOF to Momentum"""
        for yield_ in self.yields:
            yield_.convert_tof_to_momentum()

    ### Plotting ###

    def plot(self, ax=None, labels=None, xlim=None, ylim=None, sep_yield=0, lw=2,\
            linestyle='-', color=None, x_type='energy', major_minor_ticks:list=(10,5),\
            yscale=None, show_I=True, show_Up=True, show_wl=False, show_gamma=True,\
            offset_yield=0, normalize=False):
        """Plot the yield vs energy, momentum, or time"""
        ax = ax if ax else self.ax
        labels = labels if labels else self.labels

        for i, yield_ in enumerate(self.yields):
            label = labels[i]
            yield_.plot(ax=ax, label=label, sep_yield=i*sep_yield, lw=lw,\
                        linestyle=linestyle, color=color, x_type=x_type,\
                        show_I=show_I, show_Up=show_Up, show_wl=show_wl,\
                        show_gamma=show_gamma, offset_yield=i*offset_yield,\
                        normalize=normalize)

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if yscale:
            ax.set_yscale(yscale)
        if major_minor_ticks:
            major, minor = major_minor_ticks
            ax.xaxis.set_major_locator(MultipleLocator(major))
            ax.xaxis.set_minor_locator(MultipleLocator(minor))

    def label_plot(self, ax=None, xlabel=None, ylabel=None, title=None, x_unit='eV',\
                   fontsize=10, tfontsize=12, fontweight='bold'):
        """Label the plot"""
        ax = ax if ax else self.ax
        xlabel = xlabel if xlabel else f"Photoelectron Energy [{x_unit}]"
        ylabel = ylabel if ylabel else "Relative Yield [a.u.]"
        title = title if title else f"Photoelectron Yield at {self.wl[0]} nm"
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)
        ax.set_title(title, fontsize=tfontsize, fontweight='bold')

    def gen_legend(self, save_path="legend.png",save_dir=None, show_label=False, show_I=True,\
                   show_Up=True, show_wl=False, show_gamma=True, show_Ip=False,\
                   show_channel_closure=True, show_photon_energy=False,\
                    figsize=(4,5), dpi=120, tight_layout=True):
        """ Generate Lengend for the plot seperately"""
        fig, ax = plt.subplots(figsize = figsize, dpi=dpi, tight_layout=tight_layout)
        for yield_ in self.yields:
            label = yield_.gen_label(show_label=show_label,show_I=show_I, show_Up=show_Up, show_wl=show_wl,\
                                     show_gamma=show_gamma, show_Ip=show_Ip, show_channel_closure=show_channel_closure,\
                                        show_photon_energy=show_photon_energy)
            ax.plot([], label=label)
        ax.legend(loc="best", framealpha=1, frameon=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)
        if save_dir is None:
            save_dir = self.dir_path
        fig.savefig(os.path.join(save_dir, save_path))

    # Export Data
    def export_data(self, file_name=None, fn_extra=None, dir_path=None, data_type='energy', add_calibration_header=True, normalize=False, method='jacobian', E_max=None, dE=None):
        """Export data to file"""
        file_name_ed = file_name
        for i, yield_ in enumerate(self.yields):
            if fn_extra:
                file_name_ed = file_name if file_name else yield_.file_name.replace('.txt', f'_{data_type}')
                if fn_extra == 'I':
                    fn_extra_ed = f"_I{round(self.I[i],2)}TW"
                file_name_ed = file_name_ed+fn_extra_ed
            yield_.export_data(file_name=file_name_ed, dir_path=dir_path, data_type=data_type, add_calibration_header=add_calibration_header, normalize=normalize, method=method, E_max=E_max, dE=dE)


