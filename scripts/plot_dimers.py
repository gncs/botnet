import argparse
from typing import Tuple

import ase.data
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from utils import style_dict

plt.rcParams.update({'font.size': 6})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs',
                        help='name and path to XYZ configurations (name, path)',
                        action='append',
                        required=True)
    return parser.parse_args()


def parse_config_path(name_path_tuple: str) -> Tuple[str, pd.DataFrame]:
    name, path = name_path_tuple.split(',')
    atoms_list = ase.io.read(path, format='extxyz', index=':')
    return name, pd.DataFrame({
        'distance': [atoms.info['separation'] for atoms in atoms_list],
        'energy': [atoms.info['energy'] for atoms in atoms_list],
        'config_type': [atoms.info['config_type'] for atoms in atoms_list],
    })


def main():
    args = parse_args()
    predictions = [parse_config_path(path) for path in args.configs]

    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(6.0, 1.5), constrained_layout=True, sharey='row')

    config_types = ['HC', 'CC', 'CO', 'HO', 'HH', 'OO']
    for ax, config_type in zip(axes, config_types):
        for name, df in predictions:
            selection = df[df['config_type'] == config_type]
            ax.plot(selection['distance'], selection['energy'] - np.min(selection['energy']), **style_dict[name])

        ax.set_xticks([0.0, 1.5, 3.0, 4.5])
        ax.set_xlabel('Distance [Å]')

        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        ax.set_title(r'$\mathrm{' + config_type[0] + r'}-\mathrm{' + config_type[1] + '}$')

    axes[0].set_ylabel(r'$\Delta E$ [eV]')
    axes[0].legend()

    fig.show()


if __name__ == '__main__':
    main()
