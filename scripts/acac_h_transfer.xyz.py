import argparse
from typing import Tuple, List

import ase.data
import ase.io
import matplotlib.pyplot as plt
import numpy as np

from plot_settings import style_dict

fig_width = 6.0 / 3  # inches
fig_height = 2.1

plt.rcParams.update({'font.size': 6})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions',
                        help='name and path to XYZ configurations (name, path)',
                        action='append',
                        required=True)
    return parser.parse_args()


def parse_config_path(name_path_tuple: str) -> Tuple[str, List[float]]:
    name, path = name_path_tuple.split(',')
    atoms_list = ase.io.read(path, format='extxyz', index=':')
    return (
        name,
        [atoms.info['energy'] if 'energy' in atoms.info.keys() else atoms.info['energy_wB97X'] for atoms in atoms_list],
    )


def main():
    args = parse_args()
    predictions = [parse_config_path(tup) for tup in args.predictions]

    # Plot curve
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height), constrained_layout=True)

    for name, energies in predictions:
        ax.plot((energies - np.min(predictions[1][1])) * 1000, **style_dict[name])

    ax.set_ylabel(r'$\Delta E$ [meV]')

    # ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
