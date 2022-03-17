import argparse

import ase.data
import ase.io
import matplotlib.pyplot as plt
import numpy as np

from utils import style_dict

fig_width = 6.0  # inches
fig_height = 2.1

plt.rcParams.update({'font.size': 6})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to XYZ configurations', required=True)
    parser.add_argument('--dft',help='path to dft XYZ configurations')
    return parser.parse_args()


def main():
    args = parse_args()
    contributions = np.array(
        [atoms.info['contributions'] for atoms in ase.io.read(args.path, format='extxyz', index=':')])  # [c, e]
    contributions = contributions.transpose()  # [e, c]
    energies = np.array([atoms.info['energy'] for atoms in ase.io.read(args.path, format='extxyz', index=':')])  # [e]
    energies = np.expand_dims(energies, axis=0)  # [1, c]
    displacement = np.array([atoms.info['displacement'] for atoms in ase.io.read(args.path, format='extxyz', index=':')])
    dft_energy = np.array(
        [atoms.info['energy'] for atoms in ase.io.read(args.dft, format='extxyz', index=':')])  # [e]

    array = np.concatenate([energies, contributions[:]], axis=0)

    # Plot curve
    fig, axes = plt.subplots(nrows=1, 
                            ncols=array.shape[0],
                            sharey='row', 
                            figsize=(fig_width, fig_height), 
                            constrained_layout=True)

    for i, (ax, energies) in enumerate(zip(axes, array)):
        e_shift = energies[-1]
        ax.plot(displacement, energies - e_shift, **style_dict['botnet'])
        if i == 0:
            ax.plot(displacement, dft_energy - e_shift, **style_dict['dft'])
            ax.set_title(r'$E_\mathrm{tot}$' + f' - ({e_shift:.3f} eV)')
        else:
            j = i-1
            ax.set_title(rf'$E_{j}$' + f' - ({e_shift:.3f} eV)')

    axes[0].set_ylabel(r'$E$ [eV]')
    axes[0].set_xlabel('Displacement')
    axes[0].legend(bbox_to_anchor=(1.04,1), loc="upper left")

    fig.savefig('contributions.pdf')


if __name__ == '__main__':
    main()
