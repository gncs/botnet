import argparse
from typing import Tuple

import ase.data
import ase.io
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.boxplots import violinplot

from utils import style_dict

plt.rcParams.update({'font.size': 6})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', action='append', required=True)
    return parser.parse_args()


def parse_config_path(name_path_tuple: str) -> Tuple[str, np.ndarray, np.ndarray]:
    name, path = name_path_tuple.split(',')
    atoms_list = ase.io.read(path, format='extxyz', index=':')
    energies = np.array([atoms.info['energy'] for atoms in atoms_list]) * 1000  # meV
    forces = np.concatenate([atoms.arrays['forces'] for atoms in atoms_list], axis=0) * 1000  # meV / Ang
    return name, energies, forces


def main():
    args = parse_args()
    tuples = [parse_config_path(tup) for tup in args.configs]
    ref_name, ref_energies, ref_forces = tuples[0]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.0, 2.5), constrained_layout=True, sharex='col')
    names = [name for name, energies, forces in tuples[1:]]

    # Energies
    ax = axes[0]
    for idx, (name, energies, forces) in enumerate(tuples[1:]):
        violinplot(ax=ax,
                   data=[np.abs(ref_energies - energies)],
                   positions=[idx],
                   show_boxplot=False,
                   side='both',
                   plot_opts={
                       'violin_fc': style_dict[name]['color'],
                       'violin_alpha': 1.0,
                   })

    ax.set_ylabel(r'$\Delta E$ [meV]')

    # Forces
    ax = axes[1]
    for idx, (name, energies, forces) in enumerate(tuples[1:]):
        violinplot(ax=ax,
                   data=[np.linalg.norm(ref_forces - forces, axis=-1)],
                   positions=[idx],
                   show_boxplot=False,
                   side='both',
                   plot_opts={
                       'violin_fc': style_dict[name]['color'],
                       'violin_alpha': 1.0,
                   })

    axes[1].set_ylabel(r'$|\Delta F|$ [meV/Å]')

    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels([style_dict[name]['label'] for name in names])

    ax.set_xlim(left=-0.5, right=len(names) - 0.5)

    fig.savefig('errors.pdf')


if __name__ == '__main__':
    main()
