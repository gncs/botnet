import argparse
from typing import Tuple, List

import ase.data
import ase.io
import matplotlib.pyplot as plt
import numpy as np

fig_width = 6.0 / 3  # inches
fig_height = 2.1

plt.rcParams.update({'font.size': 6})

from plot_settings import style_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions',
                        help='name and path to XYZ configurations (name, path)',
                        action='append',
                        required=True)
    return parser.parse_args()


def parse_config_path(name_path_tuple: str) -> Tuple[str, List[Tuple[float, float, float]], List[float]]:
    name, path = name_path_tuple.split(',')
    atoms_list = ase.io.read(path, format='extxyz', index=':')
    return (
        name,
        [tuple(atoms.info['dihedrals']) for atoms in atoms_list],
        [atoms.info['energy'] if 'energy' in atoms.info.keys() else atoms.info['energy_wB97X'] for atoms in atoms_list],
    )


def main():
    args = parse_args()
    predictions = [parse_config_path(tup) for tup in args.predictions]

    # Plot curve
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height), constrained_layout=True)
    alpha = 131.0
    beta = 150.0

    for name, angle_tuples, energies in predictions:
        data = list(
            filter(
                lambda t: np.isclose(t[0][0], alpha) and np.isclose(t[0][1], beta),
                zip(angle_tuples, energies),
            ))

        angles = [t[0][-1] for t in data]
        energies = np.array([t[1] for t in data])

        ax.plot(angles, (energies - np.min(energies)) * 1000, **style_dict[name])

    ax.set_title(fr'$\alpha$={alpha:.1f}°, $\beta$={beta:.1f}°')
    ax.set_xticks([0, 60, 120, 180, 240, 300])
    ax.set_xlabel(r'$\gamma$ [°]')
    ax.set_ylabel(r'$\Delta E$ [meV]')

    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
