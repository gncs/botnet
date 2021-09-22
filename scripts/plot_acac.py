import argparse
from typing import Tuple

import ase.data
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import style_dict

plt.rcParams.update({'font.size': 6})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', help='path to XYZ training configurations', required=True)
    parser.add_argument('--dihedral_configs', action='append', required=True)
    parser.add_argument('--transfer_configs', action='append', required=True)
    return parser.parse_args()


def parse_dihedral_configs(name_path_tuple: str) -> Tuple[str, pd.DataFrame]:
    name, path = name_path_tuple.split(',')
    atoms_list = ase.io.read(path, format='extxyz', index=':')
    return (name,
            pd.DataFrame({
                'dihedral': [atoms.info['dihedral_angle'] for atoms in atoms_list],
                'energy': [atoms.info['energy'] * 1000 for atoms in atoms_list],
            }))


def parse_transfer_configs(name_path_tuple: str) -> Tuple[str, pd.DataFrame]:
    name, path = name_path_tuple.split(',')
    atoms_list = ase.io.read(path, format='extxyz', index=':')
    return (name,
            pd.DataFrame({
                'energy': [atoms.info['energy'] * 1000 for atoms in atoms_list],
                'd1': [atoms.get_distance(3, 11) for atoms in atoms_list],
                'd2': [atoms.get_distance(5, 11) for atoms in atoms_list],
            }))


def main():
    args = parse_args()
    training_atoms = ase.io.read(args.training, format='extxyz', index=':')
    dihedral_predictions = [parse_dihedral_configs(path) for path in args.dihedral_configs]
    transfer_predictions = [parse_transfer_configs(path) for path in args.transfer_configs]

    fig, axes = plt.subplots(nrows=2,
                             ncols=2,
                             figsize=(6.0, 2.1),
                             constrained_layout=True,
                             sharex='col',
                             gridspec_kw={'height_ratios': (5, 1)})

    # Dihedral curve
    ax = axes[0][0]
    ref_energy = np.min(dihedral_predictions[0][1]['energy'])

    for name, df in dihedral_predictions:
        ax.plot(df['dihedral'], (df['energy'] - ref_energy), **style_dict[name])

    ax.set_ylabel(r'$\Delta E$ [meV]')
    ax.legend()

    # Dihedral Histogram
    ax = axes[1][0]
    train_dihedrals = []
    for atoms in training_atoms:
        if atoms.get_dihedral(0, 1, 2, 3) < 180:
            train_dihedrals.append(atoms.get_dihedral(0, 1, 2, 3))
        else:
            train_dihedrals.append(360 - atoms.get_dihedral(0, 1, 2, 3))

    ax.hist(train_dihedrals, bins=np.arange(0, 185, 5), color='black', label='Training data')
    ax.set_xlabel('Dihedral Angle [°]')
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_ylabel('Count')
    ax.legend()

    # H transfer
    ax = axes[0][1]
    ref_energy = np.min(transfer_predictions[0][1]['energy'])
    for name, df in transfer_predictions:
        ax.plot(df['d1'], df['energy'] - ref_energy, **style_dict[name])

    # H transfer histogram
    ax = axes[1][1]
    step_size = 0.05
    d_range = np.arange(1.05, 1.50, step=step_size)
    d1s = np.array([atoms.get_distance(3, 11) for atoms in training_atoms])
    d2s = np.array([atoms.get_distance(5, 11) for atoms in training_atoms])
    train_hist = np.histogram2d(d1s, d2s, bins=d_range)[0]
    sym_hist = train_hist + train_hist.transpose()

    ref_df = transfer_predictions[0][1]
    path_hist = np.histogram2d(ref_df['d1'], ref_df['d2'], bins=d_range)[0]
    hist = sym_hist * path_hist

    ax.bar(x=0.5 * (d_range[:-1] + d_range[1:]),
           height=np.sum(hist, axis=1),
           width=step_size,
           color='black',
           label='Training data')

    ax.set_xlabel('Distance [Å]')
    ax.set_ylabel('Count')

    plt.show()


if __name__ == '__main__':
    main()
