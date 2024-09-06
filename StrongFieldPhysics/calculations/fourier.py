"""Fourier Transformation

Example for using fft:
    t = np.arange(0, 5, 0.01)
    dc_offset = 1
    signal = np.sin(2*np.pi*2*t)+np.sin(2*np.pi*5*t) + np.random.random(len(t)) + dc_offset
    freq, power_spec = fft(signal, 0.01, auto_remove_dc_offset=True)
    plt.plot(freq, power_spec);plt.show()

Example for using filter_signal:
    t = np.arange(0, 5, 0.01)
    signal = np.sin(2*np.pi*2*t)+np.sin(2*np.pi*5*t) + np.random.random(len(t))
    plt.plot(t, signal);plt.show()
    signal_filtered = filter_signal(signal, 0.01, threshold=50)
    plt.plot(t, signal_filtered.real)
    plt.plot(t, signal)
    plt.show()

"""
import numpy as np


# Fast Fourier Transform
def fft(signal_amp, dt, dc_offset=0, auto_remove_dc_offset=False, calc_phase=False, power_spectrum=False, zero_padding=0):
    """Fast Fourier Transform
    signal_amp: signal amplitude (y-axis)
    dt: time step (x-axis)
    """
    if dc_offset != 0:
        signal_amp = signal_amp - dc_offset
    elif auto_remove_dc_offset:
        signal_amp = signal_amp - np.mean(signal_amp)
    if zero_padding!=0:
        signal_amp = np.pad(signal_amp, (zero_padding, zero_padding), 'constant', constant_values=(0, 0))
    N = len(signal_amp)
    # only first half of the spectrum is useful
    N2 = N//2
    freq = 1/(N*dt) * np.arange(N2)
    y = np.fft.fft(signal_amp)[:N2]
    if power_spectrum: # power spectrum as defined in physics
        power_spectrum = np.real(y * np.conj(y) / N)
    else: # magnitude of the signal
        power_spectrum = np.real(np.abs(y))
    # TODO: Change 'power_spectrum' to 'magnitude'
    if calc_phase:
        phase = np.angle(y)
        return freq, power_spectrum, phase
    return freq, power_spectrum


def filter_signal(signal_amp, dt, threshold=None, remove_high_freq_above=None, remove_low_freq_below=None):
    """Filters the signal using FFT
    signal_amp: signal amplitude (y-axis)
    dt: time step (x-axis)
    threshold: threshold for filtering
    remove_high_freq_above: cutoff frequency

    For plotting use y.realt
    """
    N = len(signal_amp)
    # only first half of the spectrum is useful
    freq = 1/(N*dt) * np.arange(N)
    y = np.fft.fft(signal_amp)
    power_spectrum = y * np.conj(y) / N
    # filter the signal
    ind = np.ones_like(freq, dtype=bool)
    if threshold is not None:
        ind = ind & (power_spectrum > threshold)
    if remove_high_freq_above is not None:
        ind = ind & (freq < remove_high_freq_above)
    if remove_low_freq_below is not None:
        ind = ind & (freq > remove_low_freq_below)
    power_spectrum = power_spectrum * ind
    y = y * ind
    # inverse FFT
    return np.fft.ifft(y)


def correct_phase_arr(phase, tol=1E-7):
    """Correct the phase to be between 0 and 2pi instead of
    -pi to pi that you get from np.angle
    Also consider small numerical errors (<1E-7) as zero.
    """
    phase = np.where(np.abs(phase) < tol, 0, phase)
    phase = np.where(phase < 0, phase + 2*np.pi, phase)
    return phase

def correct_phase(phase, tol=1E-7):
    """Correct the phase to be between 0 and 2pi instead of
    -pi to pi that you get from np.angle
    Also consider small numerical errors (<1E-7) as zero.
    """
    if np.abs(phase) < tol:
        return 0
    if phase < 0:
        return phase + 2*np.pi
    return phase