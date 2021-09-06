import argparse
import os
from typing import Tuple, List

import ase.data
import ase.io
import matplotlib.pyplot as plt
import numpy as np

fig_width = 6.0 / 3  # inches
fig_height = 2.1

plt.rcParams.update({'font.size': 6})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', help='path to XYZ training configurations', required=True)
    parser.add_argument('--predictions', help='path to XYZ configurations', action='append', required=True)
    return parser.parse_args()


def parse_config_path(path: str) -> Tuple[str, List[ase.Atoms]]:
    basename = os.path.basename(path)
    root, ext = os.path.splitext(basename)
    name = root.split('_')[-1]
    return (
        name,
        ase.io.read(path, format='extxyz', index=':'),
    )


def main():
    args = parse_args()
    training_atoms = ase.io.read(args.training, format='extxyz', index=':')
    predictions = [parse_config_path(path) for path in args.predictions]

    # Plot curve
    fig, axes = plt.subplots(nrows=2,
                             ncols=1,
                             figsize=(fig_width, fig_height),
                             constrained_layout=True,
                             sharex='col',
                             gridspec_kw={'height_ratios': (5, 1)})

    ax = axes[0]
    for name, atoms_list in predictions:
        dihedrals = np.array([atoms.info['dihedral_angle'] for atoms in atoms_list])
        energies = np.array([atoms.info['energy'] for atoms in atoms_list])
        ax.plot(
            dihedrals,
            (energies - energies[0]) * 1000,
            color='black',
            label=name,
        )

    ax.set_ylabel(r'$\Delta E$ [meV]')
    ax.legend()

    ax = axes[1]
    train_dihedrals = []
    for atoms in training_atoms:
        if atoms.get_dihedral(0, 1, 2, 3) < 180:
            train_dihedrals.append(atoms.get_dihedral(0, 1, 2, 3))
        else:
            train_dihedrals.append(360 - atoms.get_dihedral(0, 1, 2, 3))

    ax.hist(train_dihedrals, bins=np.arange(0, 180, 5), color='black', label='Training data')
    ax.set_xlabel('Dihedral Angle [°]')
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_ylabel('Count')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
