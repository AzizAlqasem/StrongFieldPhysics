from StrongFieldPhysics.ionization.pe_yield import PE_yield
import matplotlib.pyplot as plt

def test_t2e():
    # path = r'C:\Users\alqasem.2\OneDrive - The Ohio State University\Strong Field Squad\Alkali Machine and VMI\Code\StrongFieldPhysics\StrongFieldPhysics\test\ionization\Ar_elec_800nm_TDC2228A.csv'
    path = r'/Users/aziz/Library/CloudStorage/OneDrive-TheOhioStateUniversity/Strong Field Squad/Alkali Machine and VMI/Code/StrongFieldPhysics/StrongFieldPhysics/test/ionization/Ar_elec_800nm_TDC2228A.csv'

    pe_1 = PE_yield(path)
    pe_1.load_data()
    pe_1.correct_tof()
    pe_1.convert_tof_to_energy(method='jacobian')
    pe_1.creat_figure()
    # pe_1.yeild_E
    pe_1.plot(xlim=[0, 60], yscale='log', label='jacobian')

    pe_2 = PE_yield(path, dE=0.05)
    pe_2.load_data()
    pe_2.correct_tof()
    pe_2.convert_tof_to_energy(method='integration', E_max=60, time_resolution_interp=0.1)
    # pe_2.creat_figure()
    pe_2.plot(ax=pe_1.ax, xlim=[0, 60], yscale='log', label='integration')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    test_t2e()