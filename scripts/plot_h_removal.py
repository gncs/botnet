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
        'distance': [atoms.info['OH_dist'] for atoms in atoms_list],
        'energy': [atoms.info['energy'] for atoms in atoms_list],
    })


def main():
    args = parse_args()
    predictions = [parse_config_path(path) for path in args.configs]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.0 / 3, 1.5), constrained_layout=True)

    for name, df in predictions:
        ax.plot(df['distance'], df['energy'], **style_dict[name])

    ax.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ax.set_xlabel('Distance [Å]')

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    ax.set_ylabel(r'$E$ [eV]')
    ax.legend()

    fig.show()


if __name__ == '__main__':
    main()
