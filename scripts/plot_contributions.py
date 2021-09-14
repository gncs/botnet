import argparse

import ase.data
import ase.io
import matplotlib.pyplot as plt
import numpy as np

fig_width = 6.0  # inches
fig_height = 2.1

plt.rcParams.update({'font.size': 6})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to XYZ configurations', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    contributions = np.array(
        [atoms.info['contributions'] for atoms in ase.io.read(args.path, format='extxyz', index=':')])  # [c, e]
    contributions = contributions.transpose()  # [e, c]
    energies = np.array([atoms.info['energy'] for atoms in ase.io.read(args.path, format='extxyz', index=':')])  # [c]
    energies = np.expand_dims(energies, axis=0)  # [1, c]

    array = np.concatenate([energies, contributions[1:]], axis=0)

    # Plot curve
    fig, axes = plt.subplots(nrows=1, ncols=array.shape[0], figsize=(fig_width, fig_height), constrained_layout=True)

    for i, (ax, energies) in enumerate(zip(axes, array)):
        e_min = np.min(energies)
        ax.plot(energies - e_min, color='black')
        if i == 0:
            ax.set_title(r'$E_\mathrm{tot}$' + f' - ({e_min:.3f})')
        else:
            ax.set_title(rf'$E_{i}$' + f' - ({e_min:.3f})')

    # ax.set_ylabel(r'$E$ [eV]')
    # ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
