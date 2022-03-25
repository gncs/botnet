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
    name, *paths = name_path_tuple.split(',')
    frames = pd.DataFrame([{
        'distance': atoms.info['distance'],
        'energy': atoms.info['energy'],
        'config_type': atoms.info['config_type'],
    } for path in paths for atoms in ase.io.read(path, format='extxyz', index=':')])

    df = pd.DataFrame(frames).groupby(['distance','config_type']).aggregate(
        mean_energy=pd.NamedAgg(column='energy', aggfunc='mean'),
        std_energy=pd.NamedAgg(column='energy', aggfunc='std'),
    ).reset_index()

    return name, df


def main():
    args = parse_args()
    predictions = [parse_config_path(path) for path in args.configs]

    fig, axes_grid = plt.subplots(nrows=2, ncols=3, figsize=(6.0, 3.0), constrained_layout=True)
    axes = [ax for row in axes_grid for ax in row]

    config_types = ['HC', 'CC', 'CO', 'HO', 'HH', 'OO']
    for ax, config_type in zip(axes, config_types):
        dft_ref = predictions[0][1]
        for index, (name, df) in enumerate(predictions):
            selection = df[df['config_type'] == config_type]
            ref_energy = selection['mean_energy'].iloc[-1]
            ax.plot(selection['distance'], selection['mean_energy'] - ref_energy, **style_dict[name])
            ax.fill_between(
                x=selection['distance'],
                y1=selection['mean_energy'] - selection['std_energy'] - ref_energy,
                y2=selection['mean_energy'] + selection['std_energy'] - ref_energy,
                alpha=0.3,
                zorder=2 * index,
                **style_dict[name],
        )
        ax.set_xticks([0.0, 1.5, 3.0, 4.5])
        ax.set_xlabel('Distance [Å]')

        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        ax.set_title(r'$\mathrm{' + config_type[0] + r'}-\mathrm{' + config_type[1] + '}$')

    axes[0].set_ylabel(r'$\Delta E$ [eV]')
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")

    fig.savefig('dimers.pdf')


if __name__ == '__main__':
    main()
