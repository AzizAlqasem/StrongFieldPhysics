# use NIST data for energy levels
# Reference: 'https://physics.nist.gov/PhysRefData/ASD/levels_form.html'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import os

import requests as req


def gen_url(species, z=0, format=3): 
    # species is the element name e.g. 'Cs' for Cesium
    # z is the ionization state 0 for neutral
    # format is the output format: 0 for HTML, 1 for ASCII(test), 2 for CSV, 3 for TAB-delimited
    url = 'https://physics.nist.gov/cgi-bin/ASD/energy1.pl?de=0&spectrum=' + species + '+' + str(z) + '&submit=Retrieve+Data&units=1&format=' + str(format) + '&output=0&page_size=15&multiplet_ordered=0&conf_out=on&term_out=on&level_out=on&unc_out=1&j_out=on&lande_out=on&perc_out=on&biblio=on&temp='
    return url

def download_energy_levels(species:str, z=0, save=False, filename='energy_levels'):
    url = gen_url(species, z)
    response = req.get(url)
    if response.status_code != 200:
        print(f'Error: Could not download data for {species} {z}+.')
        return None
    data = response.text
    if save:
        this_file_path = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(this_file_path, 'energy_level_data', filename + '_' + species + str(z) + '.txt')
        with open(filename, 'w') as file:
            file.write(data)
    return data

def load_energy_levels(species:str, z=0, filename='energy_levels'):
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_file_path, 'energy_level_data', filename + '_' + species + str(z) + '.txt')
    with open(filename, 'r') as file:
        data = file.read()
    return data

def parse_energy_levels(data:str, species:str):
    """ example data:
    Configuration	Term	J	Prefix	Level (eV)	Suffix	Uncertainty (eV)	Reference
    "5p6.6s"	"2S"	"1/2"	""	"0.00000000"	""	""	"L15024"
    "5p6.6p"	"2P*"	"1/2"	""	"1.385928617528"	""	"0.000000000010"	"L12151"
    "5p6.6p"	"2P*"	"3/2"	""	"1.454620692074"	""	"0.000000000025"	"L15527"
    ...
    """
    data = data.split('\n')
    data_dict = {key: [] for key in data[0].split('\t')}
    for i, line in enumerate(data[1:]):
        if line.startswith('"' + species):
            # last_line_i = i
            break
        line = line.split('\t')
        for key, value in zip(data_dict.keys(), line):
            data_dict[key].append(value.replace('"', ''))
    return data_dict


# Plot energy levels
def plot_energy_levels(data_dict, species:str, WL,  max_labels=10, ):
    energy = np.array(data_dict['Level (eV)'], dtype=float)
    Configuration = data_dict['Configuration']
    fig, ax = plt.subplots(figsize=(4, 7), tight_layout=True)
    x = [1] * len(energy)
    ax.scatter(x, energy, s=30000, marker="_", linewidth=1, color='black', alpha=1)
    old_tx = ''
    c = 0
    for xi, yi, tx in zip(x, energy, Configuration):
        if tx == old_tx or c>max_labels:
            txt = ''
        else:
            txt = tx[:2] + ' ' + tx[4:]
        ax.annotate(txt, xy=(1.023*xi, yi), xytext=(7, 5), size=10,
                    ha="center", va='top', textcoords="offset points", fontweight='bold')
        old_tx = tx
        c += 1
    
    ax.set_xlim(0.98, 1.03)
    ax.set_xticks([])
    # ax.yaxis.set_minor_locator(mpl.ticker.MaxNLocator(50))
    pe = 1239.8/3500
    ax.yaxis.set_major_locator(MultipleLocator(pe))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(which='major', axis='y', linestyle='--', color='gray', alpha=0.5)
    ax.set_title(f'Energy levels of {species} | {WL}um', fontsize=14, fontweight='bold')
    ax.set_ylabel('Energy [eV]', fontsize=15, fontweight='bold')
    # Label ground state
    ax.annotate('Ground state', xy=(1, -0.06), xytext=(9, 1), size=11, ha="center", va='top', textcoords="offset points", fontweight='bold')
    
    # vertical arros of length 0.4 ev connected from bottom to top
    pe = 1239.8/(WL*1e3)
    ip = 3.8 + pe
    for dy in np.arange(0, ip, pe):
        # ax.annotate('', xy=(1, 0+dy), xytext=(1, pe+dy), arrowprops=dict(arrowstyle='<-', lw=1.5))
        # ax.hlines(pe+dy, 0.98, 1.03, color='r', lw=0.5)
        hl = 0.2
        ax.arrow(x=1, y=0+dy, dx=0, dy=pe-hl, head_width=0.0015, head_length=hl, fc='b', ec='b', width=0.0004)
    plt.show()
    # save figure
    # this_file_path = os.path.dirname(os.path.abspath(__file__))
    # filename = os.path.join(this_file_path, 'energy_level_data', f'energy_levels_{species}{z}.png')
    # plt.savefig(filename, dpi=300)





##### Test load and parse functions #####
data = load_energy_levels('He', 0)
data_dict = parse_energy_levels(data, 'He')
plot_energy_levels(data_dict, 'He', 0.5)
# print(data_dict)

############### RUN THIS TO DOWNLOAD DATA ################
# download_energy_levels('Cs', 0, save=True)
# download_energy_levels('Na', 0, save=True)
# download_energy_levels('K', 0, save=True)
# download_energy_levels('Rb', 0, save=True)
# download_energy_levels('Li', 0, save=True)
# download_energy_levels('Mg', 0, save=True)
# download_energy_levels('Ca', 0, save=True)
# download_energy_levels('He', 0, save=True)
# download_energy_levels('Ne', 0, save=True)
# download_energy_levels('Ar', 0, save=True)
# download_energy_levels('Xe', 0, save=True)
# download_energy_levels('Kr', 0, save=True)
