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
    parser.add_argument('--configs', help='path to XYZ configurations', action='append', required=True)
    return parser.parse_args()


def parse_config_path(path: str) -> Tuple[str, List[float]]:
    basename = os.path.basename(path)
    root, ext = os.path.splitext(basename)
    name = root.split('_')[-1]
    return (
        name,
        [float(atoms.info['energy']) for atoms in ase.io.read(path, format='extxyz', index=':')],
    )


def main():
    args = parse_args()
    info = [parse_config_path(path) for path in args.configs]

    # Plot curve
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height), constrained_layout=True)

    for name, energies in info:
        ax.plot(
            (np.array(energies) - energies[0]) * 1000,
            color='black',
            label=name,
        )

    ax.set_ylabel(r'$\Delta E$ [meV]')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
