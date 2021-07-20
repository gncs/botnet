import argparse
import logging

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import torch

from e3nnff import data, tools, models

fig_width = 2.5
fig_height = 2.1

plt.rcParams.update({'font.size': 6})

colors = [
    '#1f77b4',  # muted blue
    '#d62728',  # brick red
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs_path', help='path to XYZ configurations', required=True)
    parser.add_argument('--r_max', help='distance cutoff (in Ang)', type=float, default=4.0)
    parser.add_argument('--atomic_numbers', help='atomic numbers (comma-separated)', type=str, required=True)
    parser.add_argument('--model_path', help='path to model', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    atoms_list = ase.io.read(args.configs_path, format='extxyz', index=':')
    distances = np.array([float(atoms.info['dist']) for atoms in atoms_list], dtype=float)
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    zs = [int(z) for z in args.atomic_numbers.split(',')]
    z_table = tools.get_atomic_number_table_from_zs(zs)
    logging.info(z_table)

    loader = data.get_data_loader(
        dataset=[data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max) for config in configs],
        batch_size=len(configs),
        shuffle=False,
        drop_last=False,
    )

    model: models.BodyOrderedModel = torch.load(f=args.model_path, map_location=torch.device('cpu'))
    output = model.forward(next(iter(loader)), training=False)
    energies = tools.to_numpy(output['energy'])

    # Plot dissociation curve
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height), constrained_layout=True)

    ax.plot(distances, energies, color='black')

    ax.set_ylabel('Energy [eV]')
    ax.set_xlabel('Distance [Å]')

    fig.savefig('energies.pdf')


if __name__ == '__main__':
    main()
