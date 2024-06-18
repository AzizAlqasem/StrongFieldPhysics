"""Functions used to analyze the Photoelectron spectra
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
# fit gaussian
from scipy.optimize import curve_fit
# find peaks
from scipy.signal import find_peaks

from StrongFieldPhysics.calculations.calculations import photon_energy, ponderomotive_energy, channel_closure, gamma
from StrongFieldPhysics.tools.arrays import find_indx_max_in_yarr_from_xrange, find_indx_min_in_yarr_from_xrange
from StrongFieldPhysics.calculations.fourier import fft, filter_signal, correct_phase

def gaussian(x, mean, sigma):
    # sigma = fwhm / 2.35482 #np.sqrt(8*np.log(2))
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

# A class that analyze the photoelectron spectra envelope
class PE_Envelope:

    def __init__(self, energy, yld, wavelength, intensity, IP, target=None):
        """Analyzing the photoelectron spectra envelope

        Args:
            energy (array or list): energy axis of the photoelectron spectra
            yld (array or list): yield axes of the photoelectron spectra
            wavelength (_type_): wavelength of the laser field in nm
            intensity (_type_): Intensity of the laser field in TW/cm^2
            IP (_type_): Ionization potential of the target in eV
        """
        self.energy = energy
        self.yld = yld
        self.yld_log = np.log10(yld)
        self.wavelength = wavelength
        self.intensity = intensity
        self.IP = IP
        self.target = target
        self.pe = photon_energy(wavelength)
        self.Up = ponderomotive_energy(intensity, wavelength/1e3)
        self.gamma = gamma(self.Up, IP)
        self.channel_closure = channel_closure(self.Up, IP, wavelength_nm=wavelength)
        self.dE = self.energy[1] - self.energy[0]

    # Find atis peaks and minima position and amplitude, knowing that we have a peak after one photon energy
    def find_atis(self, E_start=0, E_end=None, E_shift=0, ignore_close_range=0):
        if E_end is None:
            E_end = self.energy[-1]
        pe = photon_energy(self.wavelength)
        # whithin every photon energy, we have a peak
        self.ati_peaks = [] # (index, eng_value, amplitude)
        # self.ati_minima = [] # (index, eng_value, amplitude)
        # Each ati peak has a start and end energy (Ei, Ef)
        E_start = E_start + E_shift
        E_end = E_end + E_shift
        Ei = E_start#
        Ef = Ei + pe #
        E_max_amp = -1 # initial value such that we can find the maximum
        # E_min_amp = np.inf # initial value such that we can find the minimum
        E_max_i = E_max = 0
        # E_min_i = E_min = 0
        limit = (self.energy >= E_start) & (self.energy <= E_end)
        energy = self.energy[limit]
        yld = self.yld[limit]
        ind_offset = np.argmin(np.abs(self.energy - energy[0]))
        for i in range(len(energy)-1):
            E = energy[i]
            if Ei <= E < Ef:
                if yld[i] > E_max_amp and (yld[i]>yld[i-1] and yld[i]>yld[i+1]):
                    E_max_amp = yld[i]
                    E_max = E
                    E_max_i = i
            else:
                # ignore close ati peaks
                if ignore_close_range: # ratio of the photon energy
                    last_peak_eng = self.ati_peaks[-1][1] if len(self.ati_peaks) > 0 else -100
                    if E_max < last_peak_eng + ignore_close_range*pe:
                        print(f"WARNING! Close peak at {E_max} eV is ignored")
                        Ei = Ei + pe
                        Ef = Ef + pe
                        E_max_amp = -1
                        continue
                self.ati_peaks.append((E_max_i+ind_offset, E_max, E_max_amp))
                # self.ati_minima.append((E_min_i+ind_offset, E_min, E_min_amp))
                Ei = Ei + pe
                Ef = Ef + pe
                E_max_amp = -1
                # E_min_amp = np.inf
        return self.ati_peaks#, self.ati_minima

    def find_atis_scipy(self, E_start=0, E_end=None, E_shift=0, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=None):
        if E_end is None:
            E_end = self.energy[-1]
        pe = photon_energy(self.wavelength)
        # whithin every photon energy, we have a peak
        self.ati_peaks = []
        limit = (self.energy >= E_start) & (self.energy <= E_end)
        energy = self.energy[limit]
        yld_log = self.yld_log[limit]
        yld = self.yld[limit]
        ind_offset = np.argmin(np.abs(self.energy - energy[0]))
        peaks, _ = find_peaks(yld_log, height=height, threshold=threshold, distance=distance, prominence=prominence, width=width, wlen=wlen, rel_height=rel_height)
        for i in peaks:
            self.ati_peaks.append((i+ind_offset, energy[i], yld[i]))
        return self.ati_peaks

    def find_minima(self,): # defined as the minimum between two peaks
        self.ati_minima = []
        for i in range(len(self.ati_peaks)-1):
            Ei_ind = self.ati_peaks[i][0]
            Ef_ind = self.ati_peaks[i+1][0]
            E_min_i = np.argmin(self.yld[Ei_ind:Ef_ind]) + Ei_ind
            E_min_amp = self.yld[E_min_i]
            E_min = self.energy[E_min_i]
            self.ati_minima.append((E_min_i, E_min, E_min_amp))
        return self.ati_minima

    def correct_maxima_to_center_of_minima(self,): # correct the maxima to the center of the minima
        #new_peaks = [first_pak] + [minima_i + minima_i+1/2] + [last_peak]
        self.new_atis_peaks = [self.ati_peaks[0]]
        for i in range(len(self.ati_minima)-1):
            minima_i = self.ati_minima[i][0]
            minima_i1 = self.ati_minima[i+1][0]
            new_peak = int((minima_i + minima_i1) / 2)
            self.new_atis_peaks.append((new_peak, self.energy[new_peak], self.yld[new_peak]))
        self.new_atis_peaks.append(self.ati_peaks[-1])
        return self.new_atis_peaks

    def keep_minima(self,):# loop over minima and keee minima that do not increase as we go to the right
        minv = self.ati_minima[0][2]
        for i in range(len(self.ati_minima)):
            min_amp = self.ati_minima[i][2]
            if min_amp < minv:
                minv = min_amp
            self.ati_minima[i]= (self.ati_minima[i][0], self.ati_minima[i][1], minv)

    def fit_gaussian(self, E_start, E_end, eng_type='Up', p0=None):
        """ Fit a gaussian to the photoelectron spectra within a specific range
        """
        # eng_axis
        eng = []
        yld = []
        if eng_type == 'Up':
            E_start = E_start * self.Up
            E_end = E_end * self.Up
        for peak in self.ati_peaks:
            if peak[1] >= E_start and peak[1] <= E_end:
                eng.append(peak[1])
                yld.append(peak[2])
        if eng_type == 'Up':
            eng = np.array(eng) / self.Up
        yld = np.log10(np.array(yld))
        yld -= np.min(yld) # shift the yld to be positive
        # fit
        if p0 is None:
            p0 = [eng[np.argmax(yld)], 0.1]
        popt, pcov = curve_fit(gaussian, eng, yld, p0=p0)
        self.mean_fit, self.sigma_fit = popt
        # fwhm = np.sqrt(8*np.log(2)) * self.sigma_fit
        amp = gaussian(self.mean_fit, self.mean_fit, self.sigma_fit)
        # calculate error in mean fit
        self.mean_fit_err = np.sqrt(pcov[0,0])
        return self.mean_fit, self.sigma_fit, amp


    def cal_minima_to_peaks_amp_size(self, method='loglog', avg_two_minima=False): # virical distance between a peak and the next min
        """
        Methods: loglog, ratio, linear, linear_ratio, loglog_ratio

        """
        self.minima_peaks_size = []
        self.minima_peaks_size_eng = []
        # first ati peak has no minima before it
        E_peak_amp = self.ati_peaks[0][2] # First peak; There is no minima before the peak
        E_min_amp = self.ati_minima[0][2] # this is the minima that occurs after the peak
        if method == 'loglog':
            dist = np.log10(E_peak_amp) - np.log10(E_min_amp)
        elif method == 'loglog_ratio':
            dist = (np.log10(E_peak_amp) - np.log10(E_min_amp)) / np.log10(E_min_amp)
        elif method == 'ratio':
            dist = E_peak_amp/E_min_amp
        elif method == 'linear':
            dist = E_peak_amp - E_min_amp
        elif method == 'linear_ratio':
            dist = (E_peak_amp - E_min_amp) / E_min_amp
        self.minima_peaks_size.append(dist)
        self.minima_peaks_size_eng.append(self.ati_peaks[0][1])
        # next ati peaks
        for i in range(1, len(self.ati_minima)):
            E_min_amp = self.ati_minima[i-1][2] # before the peak
            E_peak_amp = self.ati_peaks[i][2] # the peak
            if avg_two_minima:
                E_min_amp_after = self.ati_minima[i+0][2] # after the peak
                E_min_amp = (E_min_amp_after + E_min_amp)/2 # average of the two minima
            if method == 'loglog':
                dist = np.log10(E_peak_amp) - np.log10(E_min_amp)
            elif method == 'loglog_ratio':
                dist = (np.log10(E_peak_amp) - np.log10(E_min_amp)) / np.log10(E_min_amp)
            elif method == 'ratio':
                dist = E_peak_amp/E_min_amp
            elif method == 'linear':
                dist = E_peak_amp - E_min_amp
            elif method == 'linear_ratio':
                dist = (E_peak_amp - E_min_amp) / E_min_amp
            self.minima_peaks_size.append(dist)
            self.minima_peaks_size_eng.append(self.ati_peaks[i][1])

    def find_maximum_peak_range(self, E_start, E_end, eng_type='Up'):
        """ Find the maximum peak within a specific range
        """
        if eng_type == 'Up':
            E_start = E_start * self.Up
            E_end = E_end * self.Up
        max_peak = -np.inf
        for i, peak in enumerate(self.ati_peaks):
            if peak[1] >= E_start and peak[1] <= E_end:
                if peak[2] > max_peak:
                    max_peak = peak[2]
                    max_peak_i = i
        return self.ati_peaks[max_peak_i]

    def find_minimum_peak_range(self, E_start, E_end, eng_type='Up'):
        """ Find the minimum peak within a specific range
        """
        if eng_type == 'Up':
            E_start = E_start * self.Up
            E_end = E_end * self.Up
        min_peak = np.inf
        for i, peak in enumerate(self.ati_peaks):
            if peak[1] >= E_start and peak[1] <= E_end:
                if peak[2] < min_peak:
                    min_peak = peak[2]
                    min_peak_i = i
        return self.ati_peaks[min_peak_i]

    def cal_ati_phase(self, first_ati=0, method='relative', mirrors:tuple=None, correction=1.01, mirror:tuple=None):
        """ Calculate the phase of the ATI peaks
        phase/photonEnergy = ati_peak_energy / PhotonEnergy - n < 1  , where n is the order of the ATI peak
        method: relative, absolute
        relative: phase is relative to channel closure and the ati comb
        absolute: phase is relative to channel closure
        """
        self.ati_phase = []
        self.ati_phase_eng = [] #* might be redundant but it is useful to insure it matches the chosen first_ati
        if method == 'relative':
            for n, peak in enumerate(self.ati_peaks, start=first_ati):
                phase_to_pe = peak[1] / self.pe - n
                self.ati_phase.append(phase_to_pe)
                self.ati_phase_eng.append(peak[1])
        elif method == 'relative-c': # relative with missing peaks correction
            for n, peak in enumerate(self.ati_peaks, start=first_ati):
                phase_to_pe = peak[1] / self.pe - n
                while phase_to_pe > correction:
                    phase_to_pe -= 1
                self.ati_phase.append(phase_to_pe)
                self.ati_phase_eng.append(peak[1])
        elif method == 'absolute':
            for peak in self.ati_peaks:
                phase_to_pe = (peak[1] % self.pe) / self.pe
                self.ati_phase.append(phase_to_pe)
                self.ati_phase_eng.append(peak[1])
        else:
            raise ValueError("Method should be either 'relative' or 'absolute'")
        self.ati_phase = np.array(self.ati_phase)
        self.ati_phase_eng = np.array(self.ati_phase_eng)
        if mirrors is not None: # flib > 0.5 phase to < 0.5 phase within the mirror range a, b
            for _mirr in mirrors:
                a, b = _mirr # Up
                a = a * self.Up
                b = b * self.Up
                idx1 = np.argmin(np.abs(self.ati_phase_eng - a))
                idx2 = np.argmin(np.abs(self.ati_phase_eng - b))
                self.ati_phase[idx1:idx2] = 1 - self.ati_phase[idx1:idx2]
        if mirror is not None: # any phase below or above a threshol is going to be fliped
            ths, typ = mirror # e.g., 0.5, -1 (flip above 0.5 to below 0.5)
            if typ == -1:
                self.ati_phase[self.ati_phase > ths] = 1 - self.ati_phase[self.ati_phase > ths]
            elif typ == 1:
                self.ati_phase[self.ati_phase < ths] = 1 - self.ati_phase[self.ati_phase < ths]
            else:
                raise ValueError(f"The type of the mirror should be either -1 or 1. You have entered {typ}")
        self.ati_phase_mean = np.mean(self.ati_phase)
        self.ati_phase_std = np.std(self.ati_phase)

    def estimate_ati_peak_error(self, ati_order, method="std"):
        """Estimate the error in an ATI peak
        method: std: find std from valley 1 to valley 2 weighted by the amplitude (yield)
        method: fwhm: choose one peak, then find its fwhm (using width at (peak+valley)/2) then divide by peak position.
        """
        if method == "std": # find std from valley 1 to valley 2 weighted by the amplitude (yield)
            valley1 = self.ati_minima[ati_order-1] # valey before the peak
            valley2 = self.ati_minima[ati_order] # valey after the peak
            _yld = self.yld[valley1[0]:valley2[0]]
            _eng = self.energy[valley1[0]:valley2[0]]
            mean = np.average(_eng, weights=_yld)
            std = np.sqrt(np.average((_eng - mean)**2, weights=_yld))
            error = std / mean # Detection error
            # phase error is std / photon_energy
            out = (error, mean, std)
        return out

    def estimate_ati_peak_error_all(self, method="std", divide_by=2):
        """Estimate the error in all ATI peaks
        divide_by: divide the error by 2 for the error bar when plotting
        """
        self.ati_peak_error = []
        for i in range(1, len(self.ati_peaks)-1):
            e, n, s = self.estimate_ati_peak_error(i, method)
            self.ati_peak_error.append(s) # Then you can calculate the error in the phase by dividing by the photon energy
        # First and last error is the same as the second
        self.ati_peak_error = [self.ati_peak_error[0]] + self.ati_peak_error + [self.ati_peak_error[-1]]
        return self.ati_peak_error

    def calculate_diff_yield(self, window_n_photons=1):
        n_of_bins_per_photon = self.pe / self.dE
        window = int(n_of_bins_per_photon * window_n_photons)
        yld_log_avg = np.convolve(self.yld_log, np.ones(window)/window, mode='same')
        self.yld_log_diff = self.yld_log - yld_log_avg
        return self.yld_log_diff

    def calculate_diff_yield_sq(self, offset=0):
        self.yld_log_diff_sq = (self.yld_log_diff - offset)*2 # *2 is like squaring because we will have 10**(...) when we plot
        return self.yld_log_diff_sq

    def calculate_diff_yield_sq_avg(self, window_n_photons=2, offset=0):
        n_of_bins_per_photon = self.pe / self.dE
        window = int(n_of_bins_per_photon * window_n_photons)
        self.yld_log_diff_sq_avg = np.convolve(10**self.yld_log_diff_sq-offset, np.ones(window)/window, mode='same')
        return self.yld_log_diff_sq_avg

    # Fourier Transform
    def fourier_transform(self, dE=None, dc_offset=0, auto_remove_dc_offset=True, eng_lim:tuple=None, power_spectrum=False, log=True, eng_type='Up',\
                          extract_info_n_harmonics=0, source='yld', fix_phase=True):
        """Fourier transform the signal
        extract_info_n_harmonics: extract information about the first n harmonics (frequency or 1/energy)
        the extracted info is: [freq, energy_spacing, phase_2pi, amplitude]
        source: yld or diff-yld
        fix_phase: Make energy limit start and end from and to n, m photon energy.
        """
        # Energy has to be fixed bin
        if dE is None:
            dE = self.energy[1] - self.energy[0]
        if eng_lim is None:
            eng_1, eng_2 = (0, 10*self.Up) # assuming eng_type is Up
            eng_lim = (eng_1, eng_2) # to work with fix_phase
        else:
            eng_1, eng_2 = eng_lim
            if eng_type == 'Up':
                eng_1 = eng_1 * self.Up
                eng_2 = eng_2 * self.Up
        if fix_phase:
            assert(eng_type == 'Up', "Currently 'fix_phase' Only works with energy type = 'Up'")
            # check min_energy
            min_energy = self.energy[0]
            assert((eng_lim[0] * self.Up) > min_energy, f"Minimum energy that a TDC can detect is {min_energy} eV")
            # Energy bound has to start at n photon energy such that the phase make sense.
            # Thuse we slightly shift it to the nearest n Photon energy.
            eng_1 = np.round((eng_lim[0] * self.Up)/ self.pe) * self.pe
            eng_2 = np.round((eng_lim[1] * self.Up)/ self.pe) * self.pe
        idx1 = np.argmin(np.abs(self.energy - eng_1))
        idx2 = np.argmin(np.abs(self.energy - eng_2))
        if source == 'yld':
            if log:
                spectrum = self.yld_log[idx1:idx2]
            else:
                spectrum = self.yld[idx1:idx2]
        elif source == 'diff-yld':
            spectrum = self.yld_log_diff[idx1:idx2]
        else:
            raise ValueError("The source should be either 'yld' or 'diff-yld'")
        self.fft_freq, self.power_spec, self.fft_phase = fft(spectrum, dt=dE, dc_offset=dc_offset, auto_remove_dc_offset=auto_remove_dc_offset, calc_phase=True, power_spectrum=power_spectrum)
        if extract_info_n_harmonics > 0:
            self.fft_freq_phase_extracted_info_list = []
            for i in range(1, extract_info_n_harmonics+1):
                freq_i = 0.85 * i / self.pe
                freq_f = 1.15 * i / self.pe
                # search for the frequency between freq_i and freq_f
                idx = find_indx_max_in_yarr_from_xrange(self.fft_freq, self.power_spec, freq_i, freq_f)
                phase_2pi = correct_phase(self.fft_phase[idx]) / (2*np.pi) # from 0 to 1
                freq = self.fft_freq[idx]
                energy_spacing = 1/freq
                amp = np.real(self.power_spec[idx])
                self.fft_freq_phase_extracted_info_list.append((freq, energy_spacing, phase_2pi, amp))
        return self.fft_freq, self.power_spec

    def get_phase_at_freq(self, freq, phase_0_2pi=True):
        """Get phase at a given frequency"""
        idx = np.argmin(np.abs(self.fft_freq - freq))
        phase = self.fft_phase[idx]
        if phase_0_2pi:
            phase = correct_phase(phase)

    #Plot
    def plot(self, xlim=None, ylim=None, yscale='log', xlabel='Photoelectron Energy [Up]', ylabel='Normalized Yield (a.u.)', title=None,
             ax=None, fig=None, dpi=180, figsize=(6,4), vlines:list=[], anotat=False, eng_type='Up', plot_gaussian=False, p0=None, fit_min=4,\
            fit_max=8, fit_shift=8, add_phase=False, phase_txt_x=0.01, title_extra='', minor_major=(1, 0.2), dual_axis=False, fig_title=False):
        if title is None:
            title = f"Photoelectron Spectrum of {self.target} at {self.wavelength:.0f}nm and {self.intensity:.0f}TW/cm$^2${title_extra}"
        if ax is None:
            self.fig, self.ax = plt.subplots(dpi=dpi, figsize=figsize, tight_layout=True)
            ax = self.ax
            fig = self.fig
        if dual_axis:
            x2 = self.energy
            x = x2 / self.Up # Up is the main axis for backward compatibility with the gaussian fit and vlines
        elif eng_type == 'Up':
            x = self.energy / self.Up
        elif eng_type == 'eV':
            x = self.energy
        if dual_axis:
            ax2 = ax.twiny()
            ax2.set_xlabel('Photoelectron Energy [eV]', fontsize=12, weight='bold')
            ax2.xaxis.set_major_locator(MultipleLocator(10))
            ax2.xaxis.set_minor_locator(MultipleLocator(2))
            ax2.plot(x2, self.yld, alpha=0)
            if xlim is not None:
                ax2.set_xlim(np.array(xlim)*self.Up)
        ax.plot(x, self.yld, label=f'Up={round(self.Up, 2)}eV | $\gamma$={round(self.gamma, 2)} | n={round(self.channel_closure, 2)}')
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, fontsize=12, weight='bold')
        ax.set_ylabel(ylabel, fontsize=12, weight='bold')
        if fig_title:
            fig.suptitle(title, fontsize=12, weight='bold')
        else: # axis title
            ax.set_title(title, fontsize=12, weight='bold')
        if minor_major is not None:
            major, minor = minor_major
            ax.xaxis.set_major_locator(MultipleLocator(major))
            ax.xaxis.set_minor_locator(MultipleLocator(minor))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        for vline in vlines:
            xpos, label, vcolor = vline
            idx = np.argmin(np.abs(x - xpos))
            vlin_max = self.yld[idx]
            print(f"{label} {x[idx]} {eng_type} | Amp = {vlin_max}")
            label = label + f'{round(x[idx], 2)}{eng_type} | Amp={vlin_max:.2e}'.replace('e-0','E-')
            vlin_min = ax.yaxis.get_data_interval()[0]# min of y axis
            ax.vlines(x=x[idx], ymin=vlin_min, ymax=vlin_max, label=label, color=vcolor, linestyle='--')
        if anotat: # add text to the figure with information about gamma and channel closure
            ax.text(0.3, 0.95, f'Up = {round(self.Up, 3)} eV', fontsize=10, weight='bold', transform=ax.transAxes)
            ax.text(0.3, 0.90, f'Gamma = {round(self.gamma, 2)}', fontsize=10, weight='bold', transform=ax.transAxes)
            ax.text(0.3, 0.85, f'n = {round(self.channel_closure, 2)}', fontsize=10, weight='bold', transform=ax.transAxes)
            print(f'Up = {round(self.Up, 3)} eV')
            print(f'Gamma = {round(self.gamma, 3)}')
            print(f'n = {round(self.channel_closure, 3)}')
        if plot_gaussian:
            mean, sigma, amp = self.fit_gaussian(fit_min, fit_max, eng_type=eng_type, p0=p0)
            # mean_idx = np.argmin(np.abs(x - mean))
            # amp = self.yld[mean_idx] # not working because it can be at valley
            x_fit = np.linspace(fit_min, fit_max, 100)
            ax.plot(x_fit, 10**(gaussian(x_fit, mean, sigma)+fit_shift), 'g--', label=fr'Gaussian Fit | ({round(mean,2)}$\pm${self.mean_fit_err:.2f}) {eng_type}')
            print(f'Gaussian Fit | ({mean}+/-{self.mean_fit_err}) {eng_type} | Amp={round(amp, 2)}')
        if add_phase:
            err = abs(self.ati_phase_std/self.ati_phase_mean *100)
            ax.text(phase_txt_x, 0.05, f'Phase = {round(self.ati_phase_mean, 3)} | $\sigma$ = {round(self.ati_phase_std, 3)} [{round(err, 1)}%]', transform=ax.transAxes)
        return ax

    # Plot the atis peaks and minima
    def plot_env(self, ax=None, p_color='r', m_color='b', eng_type='Up', plot_minima=True, marker='o', markersize=30):
        if ax is None:
            ax = self.ax
        for peak in self.ati_peaks:
            eng = peak[1]
            if eng_type == 'Up':
                eng = eng / self.Up
            ax.scatter(eng, peak[2], color=p_color, marker=marker, s=markersize)
            #Q. what is the equvalent of markersize in scatter plot? #A. s
        if plot_minima:
            for minima in self.ati_minima:
                eng = minima[1]
                if eng_type == 'Up':
                    eng = eng / self.Up
                ax.scatter(eng, minima[2], color=m_color, marker=marker, s=markersize)
        return ax

    def vlines_photons(self, ax, n, shift=0, lw=0.5, alpha=0.5):
        # vlines = []
        for i in range(n):
            vl = (i+1) * self.pe + shift
            vl = vl / self.Up
            ax.vlines(x=vl, ymin=0, ymax=1, color='gray', linestyle='--', alpha=alpha, lw=lw)


    def plot_env_size(self, ax=None, dpi=180, figsize=(6,4), interp=True, xlabel='Photoelectron Energy [Up]',\
                      ylabel='Enhancement Size', title=None, tit_ext='', lin_sty='o', yscale='log',\
                      color='#1f77b4', interp_color='#ff7f0e', vlines:list=[], interp_kind='cubic', anotat=True,
                      max_diff=1, anotat_xpos=0.4, vmin_offset=0, major_minor=(1,0.2), y_major_minor=(2, 0.5)):
        """ Plot Enhamcement versus
        vlines: ['min'/'max', from, to, 'label']
        """
        if title is None:
            title = f"Enhancement of {self.target} at {round(self.wavelength,0)} nm and {round(self.intensity,0)}TW/cm$^2$" + tit_ext
        if ax is None: # create a new figure
            self.fig_size, self.ax_size = plt.subplots(dpi=dpi, figsize=figsize, tight_layout=True)
            ax = self.ax_size
            ax.set_xlabel(xlabel, fontsize=12, weight='bold')
            ax.set_ylabel(ylabel, fontsize=12, weight='bold')
            ax.set_title(title, fontsize=12, weight='bold')
            ax.set_yscale(yscale)
        # first_ati = 1 #* !!Warning!! if find_atis(E_start = is not zero) then first_ati should not be 1
        # self.ati_order = np.arange(first_ati, len(self.minima_peaks_size)+first_ati)
        self.ati_eng_up = np.array(self.minima_peaks_size_eng) / self.Up
        ax.plot(self.ati_eng_up, self.minima_peaks_size, lin_sty, color=color)
        major, minor = major_minor
        ax.xaxis.set_major_locator(MultipleLocator(major))
        ax.xaxis.set_minor_locator(MultipleLocator(minor))
        ymajor, yminor = y_major_minor
        ax.yaxis.set_major_locator(MultipleLocator(ymajor))
        ax.yaxis.set_minor_locator(MultipleLocator(yminor))
        if interp:
            f = interp1d(self.ati_eng_up, self.minima_peaks_size, kind=interp_kind)
            xnew = np.linspace(self.ati_eng_up[0], self.ati_eng_up[-1], 1000)
            ax.plot(xnew, f(xnew), '-', color=interp_color)
        # vlines:
        vlin_min = ax.yaxis.get_data_interval()[0] -  vmin_offset # min of y axis
        self.env_size_extracted_info_dict = {}
        for vline in vlines:
            typ, start, end, label = vline
            if typ.startswith('node'):
                if interp:
                    idx = find_indx_min_in_yarr_from_xrange(xnew, f(xnew), start, end, max_diff=max_diff)
                    eng_pos = xnew[idx]
                    amp = f(xnew)[idx]
                else:
                    idx = find_indx_min_in_yarr_from_xrange(self.ati_eng_up, self.minima_peaks_size, start, end, max_diff=max_diff)
                    eng_pos = self.ati_eng_up[idx]
                    amp = self.minima_peaks_size[idx]
                label = label + f'{eng_pos:.02f}Up | Amp={amp:.02f}'
                color = 'b'
            elif typ.startswith('hump'):
                if interp:
                    idx = find_indx_max_in_yarr_from_xrange(xnew, f(xnew), start, end, max_diff=max_diff)
                    eng_pos = xnew[idx]
                    amp = f(xnew)[idx]
                else:
                    idx = find_indx_max_in_yarr_from_xrange(self.ati_eng_up, self.minima_peaks_size, start, end, max_diff=max_diff)
                    eng_pos = self.ati_eng_up[idx]
                    amp = self.minima_peaks_size[idx]
                label = label + f'{eng_pos:0.2f}Up | Amp={amp:0.2f}'
                color = 'r'
            vlin_max = amp #self.minima_peaks_size[idx]
            ax.vlines(x=eng_pos, ymin=vlin_min, ymax=vlin_max, label=label, color=color, linestyle='--')
            print(f"{typ} at {eng_pos} Up | Amp = {vlin_max}") # for Debugging
            self.env_size_extracted_info_dict[typ] = (eng_pos, amp) # for logging
        if anotat: # add text to the figure with information about gamma and channel closure
            ax.text(anotat_xpos, 0.95, f'Up = {round(self.Up, 2)} eV', fontsize=10, weight='bold', transform=ax.transAxes)
            ax.text(anotat_xpos, 0.90, f'Gamma = {round(self.gamma, 2)}', fontsize=10, weight='bold', transform=ax.transAxes)
            ax.text(anotat_xpos, 0.85, f'n = {round(self.channel_closure, 2)}', fontsize=10, weight='bold', transform=ax.transAxes)
            print(f'Up = {round(self.Up, 3)} eV')
            print(f'Gamma = {round(self.gamma, 3)}')
            print(f'n = {round(self.channel_closure, 3)}')
        return ax

    def plot_ati_phase(self, ax=None, dpi=180, figsize=(6,4), xlabel='Photoelectron Energy [Up]', ylabel='Offset Phase',\
                        title=None, eng_type='Up', title_extra="", marker='o', linestyle='-', major_minor=(1, 0.2),\
                        ymajor_minor=(0.2, 0.1), plt_hline=True, plt_error=False, color='b', label=None, markersize=7, capsize=6,
                        pre_label='', suff_label='', ylim=None, elinewidth=None, capthick=None, title_fontsize=12, plt_hphoton=False,
                        max_err=0.3, twin_phase=False, hlcolor='r', hllabel=None, hlpos=None, skip=None ):
        if title is None:
            title = f"ATI Phase of {self.target} at {int(self.wavelength)}nm and {int(self.intensity)}TW/cm$^2$"
        if ax is None:
            self.fig_phase, self.ax_phase = plt.subplots(dpi=dpi, figsize=figsize, tight_layout=True)
            ax = self.ax_phase
            ax.set_xlabel(xlabel, fontsize=12, weight='bold')
            ax.set_ylabel(ylabel, fontsize=12, weight='bold')
            ax.set_title(title + title_extra, fontsize=title_fontsize, weight='bold')

        x = np.array(self.ati_phase_eng)[::skip]
        ati_phase = np.array(self.ati_phase)[::skip]
        if eng_type == 'Up':
            x = x / self.Up
        # if mirror is not None: # flib 0.8 phase to 0.2 within the mirror range a, b
        #     a, b = mirror # energy in Up
        #     idx1 = np.argmin(np.abs(x - a))
        #     idx2 = np.argmin(np.abs(x - b))
        #     ati_phase[idx1:idx2] = 1 - ati_phase[idx1:idx2]
        if label is None:
            label = f'{pre_label}Up={round(self.Up, 2)}eV | $\gamma$={round(self.gamma, 2)} | n={round(self.channel_closure, 2)}{suff_label}'
        if plt_error:
            err = np.array(self.ati_peak_error)/self.pe
            err[err > max_err] = max_err
            ax.errorbar(x, ati_phase, yerr=err[::skip], fmt=marker, markersize=markersize, label=label, \
                        capsize=capsize, color=color, elinewidth=elinewidth, capthick=capthick)
        else:
            ax.plot(x, ati_phase, marker=marker, linestyle=linestyle, label=label) #* warning: the x-axis is not the ATI order but array size
        # add mean and std to the plot as horizontal lines
        if twin_phase:
            self.ax_phase2 = ax.twinx()
            self.ax_phase2.set_ylabel('Offset Energy [eV]', fontsize=12, weight='bold')
            self.ax_phase2.plot(x, ati_phase*self.pe, alpha=0)
            if ylim is not None:
                self.ax_phase2.set_ylim(ylim[0]*self.pe, ylim[1]*self.pe)
            _mjt = round(self.pe/5, 1) # major tick
            _mit = round(_mjt/2, 2) # minor tick
            self.ax_phase2.yaxis.set_major_locator(MultipleLocator(_mjt))
            self.ax_phase2.yaxis.set_minor_locator(MultipleLocator(_mit))
        err = abs(self.ati_phase_std/self.ati_phase_mean *100)
        print(f'Mean = {round(self.ati_phase_mean, 5)} | $\sigma$ = {round(self.ati_phase_std, 5)} [{round(err, 3)}%]')
        if plt_hline:
            if hlcolor is None: # get the ax color
                hlcolor = ax.get_lines()[-1].get_color()
            if hllabel is None:
                hllabel = f'Mean = {round(self.ati_phase_mean, 3)} | $\sigma$ = {round(self.ati_phase_std, 3)} [{round(err, 1)}%]'
            if hlpos is None:
                hlpos = self.ati_phase_mean
            ax.axhline(y=hlpos, color=hlcolor, linestyle='--', label=hllabel)
        # x and y ticks
        major, minor = major_minor
        ax.xaxis.set_major_locator(MultipleLocator(major))
        ax.xaxis.set_minor_locator(MultipleLocator(minor))
        ymajor, yminor = ymajor_minor
        ax.yaxis.set_major_locator(MultipleLocator(ymajor))
        ax.yaxis.set_minor_locator(MultipleLocator(yminor))
        if ylim is not None:
            ax.set_ylim(ylim)
        if plt_hphoton:
            ax.axhline(y=0.5, color='gray', linestyle='--', lw=0.5)
        return ax

    def plot_diff_yield(self, ax=None, dpi=180, figsize=(6,4), xlabel='Photoelectron Energy [{}]', ylabel='Differential Yield [a.u.]',\
                        title=None, eng_type='Up', title_extra="", major_minor_ticks=None, ylim=None, xlim=None,\
                        lw=2, label=None, sep=0, yscale='log', ytype='diff', normalize=False, label_pre='', title_size=12):
        """Plot the difference in the yield
        ytype: diff, diff_sq and diff_sq_avg
        """
        if title is None:
            title = f"Differential Yield of {self.target} at {int(self.wavelength)}nm and {int(self.intensity)}TW/cm$^2$" + title_extra
        if label is None:
            label = f'Up={round(self.Up, 2)}eV | $\gamma$={round(self.gamma, 1)} | n={round(self.channel_closure, 2)}'
        if ax is None:
            self.fig_diff, self.ax_diff = plt.subplots(dpi=dpi, figsize=figsize, tight_layout=True)
            ax = self.ax_diff
            ax.set_xlabel(xlabel.format(eng_type), fontsize=12, weight='bold')
            ax.set_ylabel(ylabel, fontsize=12, weight='bold')
            ax.set_title(title, fontsize=title_size, weight='bold')
            if major_minor_ticks is None:
                if eng_type.lower() == 'up':
                    major_minor_ticks = (1, 0.2)
            #     elif eng_type.lower() == 'ev':
            #         major_minor_ticks = (20, 5)
            if major_minor_ticks is not None:
                major, minor = major_minor_ticks
                ax.xaxis.set_major_locator(MultipleLocator(major))
                ax.xaxis.set_minor_locator(MultipleLocator(minor))
            if ylim is not None:
                ax.set_ylim(ylim)
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.set_yscale(yscale)
        if eng_type.lower() == 'up':
            x = self.energy / self.Up
        elif eng_type.lower() == 'ev':
            x = self.energy
        if ytype == 'diff':
            y = 10**(self.yld_log_diff+sep) if sep != 0 else 10**self.yld_log_diff
        elif ytype == 'diff_sq':
            y = 10**(self.yld_log_diff_sq+sep) if sep != 0 else 10**self.yld_log_diff_sq
        elif ytype == 'diff_sq_avg':
            y = self.yld_log_diff_sq_avg + sep if sep != 0 else self.yld_log_diff_sq_avg
        if normalize == True:
            y = y / np.max(y)
        ax.plot(x, y, lw=lw, label=label_pre + label)
        return ax

    def gen_separate_legend(self, ax, dpi=180, figsize=(6,4), fontsize=9, title_fontsize=12, title=None, savepath=None):
        """Generate a separate legend outside the plot (different plot)"""
        handles, labels = ax.get_legend_handles_labels()
        fig, ax2 = plt.subplots(figsize=figsize, dpi=dpi)
        ax2.legend(handles, labels, loc='center', fontsize=fontsize, title=title, title_fontsize=fontsize, frameon=False, framealpha=1)
        ax2.axis('off')
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.set_frame_on(False)
        if savepath is not None:
            fig.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='w')
        return ax2

    def plot_up_at_ax(self, ax, eng_ev, y, color='gray', label=None, lw=1, alpha=1, linestyle='--', annotate=False):
        """
        When plotting yield at abs energy axis with seperate yields verticaly separated, knowing the Up is useful, and this function
        plot a line with y-points coressponds to one yield at Up, and the x axis is abs energy.
        eng_ev: energy in eV that corresponds to the n*Up
        """
        ax.plot(eng_ev, y, color=color, label=label, lw=lw, alpha=alpha, linestyle=linestyle)
        # anotitle with a label such that it is rotated and placed beside the line
        # rotation = np.arctan2(y[-1]-y[0], eng_ev[-1]-eng_ev[0]) * 180/np.pi # angle in degree
        # if annotate:
        #     ax.annotate(label, (eng_ev[0], y[0]), textcoords=label, xytext=(0,10), ha='center', va='center', rotation=rotation)
        # return ax

    # Plot fourier transform
    def plot_fft(self, ax=None, dpi=180, figsize=(6,4), xlabel='1/Energy [$eV^{-1}$]', ylabel='DFT(Yield)', title=None,\
                xlim=None, ylim=None, title_extra="", major_minor_ticks=(0.2, 0.1), lw=2, vlines:list=[], plot_info=True,\
                interp=False, freq_n=100, plot_type='plot', color='b'):
        """Plot the Fourier Transform of the signal,
        vlines: [from, to, 'label']
        plot_type: plot or stem
        """
        if title is None:
            title = f"Fourier Transform of {self.target} Spectrum at {int(self.wavelength)}nm and {int(self.intensity)}TW/cm$^2$" + title_extra
        if ax is None:
            self.fig_fft, self.ax_fft = plt.subplots(dpi=dpi, figsize=figsize, tight_layout=True)
            ax = self.ax_fft
            ax.set_xlabel(xlabel, fontsize=12, weight='bold')
            ax.set_ylabel(ylabel, fontsize=12, weight='bold')
            ax.set_title(title, fontsize=12, weight='bold')
        if interp:
            f = interp1d(self.fft_freq, self.power_spec, kind='cubic')
            self.fft_freq = np.linspace(self.fft_freq[0], self.fft_freq[-1], freq_n)
            self.power_spec = f(self.fft_freq)
        if plot_type == 'plot':
            ax.plot(self.fft_freq, self.power_spec, lw=lw)
        elif plot_type == 'stem':
            ax.stem(self.fft_freq, self.power_spec, color)# markerfmt=' ', basefmt=' ')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if major_minor_ticks is not None:
            major, minor = major_minor_ticks
            ax.xaxis.set_major_locator(MultipleLocator(major))
            ax.xaxis.set_minor_locator(MultipleLocator(minor))
        for vline in vlines:
            f, t, label = vline
            idx = find_indx_max_in_yarr_from_xrange(self.fft_freq, self.power_spec, f, t)
            phase_2pi = correct_phase(self.fft_phase[idx]) / (2*np.pi)
            energy_spacing = 1/self.fft_freq[idx]
            vlin_max = np.real(self.power_spec[idx])
            label = label + f'Eng={energy_spacing:.2f}ev | Phase={phase_2pi:.2f}[2$\pi$] | Amp={vlin_max:.2f}'
            ax.vlines(x=self.fft_freq[idx], ymin=0, ymax=vlin_max, label=label, color='r', linestyle='--', lw=1)
        if plot_info:
            for extracted_info in self.fft_freq_phase_extracted_info_list:
                freq, energy_spacing, phase_2pi, amp = extracted_info
                label = f'Eng={energy_spacing:.2f}ev | Phase={phase_2pi:.2f}[2$\pi$] | Amp={amp:.2f}'
                ax.vlines(x=freq, ymin=0, ymax=amp, label=label, linestyle='--', lw=1,) #, color='r')
        return ax

    def savefig(self, filename, fig=None, **kwargs):
        if fig is None:
            fig = self.fig
        fig.savefig(filename, **kwargs)

    # Tools

    def energy_range_cond(self, E_start, E_end, eng_type='Up'):
        """Limit the data to a specific range
        This is useful when we want remove high energy noise.
        """
        if eng_type == 'Up':
            E_start = E_start * self.Up
            E_end = E_end * self.Up
        return (self.energy >= E_start) & (self.energy <= E_end)





## Default color cycle of matplotlib
#1f77b4 (muted blue)
#ff7f0e (bright orange)
#2ca02c (muted green)
#d62728 (red)
#9467bd (dark purple)
#8c564b (brown)
#e377c2 (light purple)
#7f7f7f (dark gray)
#bcbd22 (yellow-green)
#17becf (blue)