import argparse
from typing import Tuple

import ase.data
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.utils import style_dict

plt.rcParams.update({'font.size': 6})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', action='append', required=True)
    return parser.parse_args()


def parse_config_path(name_path_tuple: str) -> Tuple[str, pd.DataFrame]:
    name, *paths = name_path_tuple.split(',')

    frames = [
        {
            'alpha': atoms.info['dihedrals'][0],
            'beta': atoms.info['dihedrals'][1],
            'gamma': atoms.info['dihedrals'][2],
            'energy': atoms.info['energy'] * 1000,  # meV
        } for path in paths for atoms in ase.io.read(path, format='extxyz', index=':')
    ]

    df = pd.DataFrame(frames).groupby(['alpha', 'beta', 'gamma']).aggregate(
        mean_energy=pd.NamedAgg(column='energy', aggfunc='mean'),
        std_energy=pd.NamedAgg(column='energy', aggfunc='std'),
    ).reset_index()

    return name, df


def main():
    args = parse_args()
    predictions = [parse_config_path(tup) for tup in args.configs]

    alpha_betas = [
        (71, 120),
        (67, 150),
        (151, 180),
    ]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6.0, 2.1), constrained_layout=True)

    for ax, (alpha, beta) in zip(axes, alpha_betas):
        ref = predictions[0][1]
        ref_energy = ref.loc[np.isclose(ref['alpha'], alpha) & np.isclose(ref['beta'], beta), 'mean_energy'].min()

        for index, (name, df) in enumerate(predictions):
            selection = df.loc[np.isclose(df['alpha'], alpha) & np.isclose(df['beta'], beta)]
            ax.plot(selection['gamma'], selection['mean_energy'] - ref_energy, zorder=2 * index + 1, **style_dict[name])
            ax.fill_between(
                x=selection['gamma'],
                y1=selection['mean_energy'] - ref_energy - selection['std_energy'],
                y2=selection['mean_energy'] - ref_energy + selection['std_energy'],
                alpha=0.3,
                zorder=2 * index,
                **style_dict[name],
            )

        ax.set_title(fr'$\alpha$={alpha:.1f}°, $\beta$={beta:.1f}°')
        ax.set_xticks([0, 60, 120, 180, 240, 300])
        ax.set_xlabel(r'$\gamma$ [°]')

    axes[0].set_ylabel(r'$\Delta E$ [meV]')
    axes[0].legend().set_zorder(2 * len(predictions))

    fig.savefig('3bpa.pdf')


if __name__ == '__main__':
    main()
