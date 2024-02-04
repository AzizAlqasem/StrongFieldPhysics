from StrongFieldPhysics.ionization.pe_yield import PE_yield
import matplotlib.pyplot as plt
import numpy as np

# path = r'C:\Users\alqasem.2\OneDrive - The Ohio State University\Strong Field Squad\Alkali Machine and VMI\Code\StrongFieldPhysics\StrongFieldPhysics\test\ionization\Ar_elec_800nm_TDC2228A.csv'
path = r'/Users/aziz/Library/CloudStorage/OneDrive-TheOhioStateUniversity/Strong Field Squad/Alkali Machine and VMI/Code/StrongFieldPhysics/StrongFieldPhysics/test/ionization/Ar_elec_800nm_TDC2228A.csv'

def test_fft():
    # pe_1 = PE_yield(path)
    # pe_1.load_data()
    # pe_1.correct_tof()
    # pe_1.convert_tof_to_energy(method='jacobian', normalize=False)
    # pe_1.creat_figure()
    # # pe_1.yeild_E
    # pe_1.plot(xlim=[0, 60], yscale='linear', label='jacobian')

    pe = PE_yield(path, dE=0.05)
    pe.load_data(num_of_bins=2038)
    pe.correct_tof()
    pe.convert_tof_to_energy(method='integration', normalize=False, E_max=70)
    # shift x axis to zero
    pe.energy = pe.energy[200:]
    pe.yeild_E = pe.yeild_E[200:]
    # pe.yeild_E = np.sin(pe.energy*2*np.pi *0.05)
    pe.creat_figure()
    pe.fourier_transform(x_type='energy', auto_remove_dc_offset=True)
    pe.plot(xlim=[0, 60], yscale='linear', label='integration', x_type='fft-e', linestyle='solid', plot_type='stem-phase')
    # pe.plot(xlim=[0, 60], yscale='linear', label='integration', x_type='energy', linestyle='solid')
    # plt.legend()
    plt.show()
    print(pe.get_phase_at_freq(1/1.2))


def test_filter_noise():
    pe0 = PE_yield(path, dE=0.05)
    pe0.load_data(num_of_bins=2038)
    pe0.correct_tof()
    pe0.convert_tof_to_energy(method='integration', normalize=False, E_max=70)

    pe = PE_yield(path, dE=0.05)
    pe.load_data(num_of_bins=2038)
    pe.correct_tof()
    pe.convert_tof_to_energy(method='integration', normalize=False, E_max=70)
    # shift x axis to zero
    # pe.energy = pe.energy[200:]
    # pe.yeild_E = pe.yeild_E[200:]
    # pe.yeild_E = np.sin(pe.energy*2*np.pi *0.05)
    pe.filter_noise(threshold=1000, remove_high_freq_above=None, remove_low_freq_below=None)
    pe.creat_figure()
    pe.fourier_transform(x_type='energy', auto_remove_dc_offset=True)
    # pe.plot(xlim=[0, 60], yscale='linear', label='integration', x_type='fft', linestyle='solid')
    pe.plot(xlim=[0, 60], yscale='linear', label='integration', x_type='energy', linestyle='solid')
    pe0.plot(ax=pe.ax,xlim=[0, 60], yscale='linear', label='integration', x_type='energy', linestyle='solid')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test_fft()
    # test_filter_noise()