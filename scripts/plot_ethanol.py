import argparse
from typing import Tuple

import ase.data
import ase.io
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

from utils import style_dict

plt.rcParams.update({'font.size': 6})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--removal_configs', action='append', required=True)
    parser.add_argument('--soft_configs', action='append', required=True)
    parser.add_argument('--hard_configs', action='append', required=True)
    return parser.parse_args()


def parse_removal_configs(parse_removal_path: str) -> Tuple[str, pd.DataFrame]:
    name, path = parse_removal_path.split(',')
    atoms_list = ase.io.read(path, format='extxyz', index=':')
    return name, pd.DataFrame({
        'distance': [atoms.info['OH_dist'] for atoms in atoms_list],
        'energy': [atoms.info['energy'] for atoms in atoms_list],
    })


def parse_displacement_path(name_path_tuple: str) -> Tuple[str, pd.DataFrame]:
    name, path = name_path_tuple.split(',')
    atoms_list = ase.io.read(path, format='extxyz', index=':')
    return name, pd.DataFrame({
        'displacement': [atoms.info['displacement'] for atoms in atoms_list],
        'energy': [atoms.info['energy'] for atoms in atoms_list],
    })


def main():
    args = parse_args()
    removal_predictions = [parse_removal_configs(path) for path in args.removal_configs]
    soft_predictions = [parse_displacement_path(path) for path in args.soft_configs]
    hard_predictions = [parse_displacement_path(path) for path in args.hard_configs]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6.0, 1.5), constrained_layout=True, sharey='row')

    # Removal
    ax = axes[0]
    for name, df in removal_predictions:
        ax.plot(df['distance'], df['energy'], **style_dict[name])

    ax.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ax.set_xlabel('Distance [Å]')

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    ax.set_title(r'Hydrogen Abstraction')
    ax.set_ylabel(r'$E$ [eV]')
    ax.legend()

    # Soft
    ax = axes[1]
    for name, df in soft_predictions:
        ax.plot(df['displacement'], df['energy'], **style_dict[name])

    ax.set_title(r'$\tilde{\nu} = 874 \, \mathrm{cm}^{-1}$')
    ax.set_xticks([-1.0, 0.0, 1.0, 2.0])
    ax.set_xlabel('Displacement')

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    # Hard
    ax = axes[2]
    for name, df in hard_predictions:
        ax.plot(df['displacement'], df['energy'], **style_dict[name])

    ax.set_title(r'$\tilde{\nu} = 3005 \, \mathrm{cm}^{-1}$')
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5])
    ax.set_xlabel('Displacement')

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    fig.savefig('ethanol.pdf')


if __name__ == '__main__':
    main()
