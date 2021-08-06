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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs_path', help='path to XYZ configurations', required=True)
    parser.add_argument('--atomic_numbers', help='atomic numbers (comma-separated)', type=str, required=True)
    parser.add_argument('--model_path', help='path to model', required=True)
    parser.add_argument('--r_max', help='distance cutoff (in Ang)', type=float, default=4.0)
    return parser.parse_args()


def main():
    args = parse_args()

    atoms_list = ase.io.read(args.configs_path, format='extxyz', index=':')
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
    ref_energies = np.array([float(atoms.info.get('DFT_energy')) for atoms in atoms_list], dtype=float)
    min_energy = np.min(ref_energies)
    assert all(energy is not None for energy in ref_energies)

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

    # Plot curve
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width, fig_height), constrained_layout=True)

    ax.plot(
        (ref_energies - min_energy) * 1000,
        color='black',
    )

    ax.plot(
        (energies - min_energy) * 1000,
        color='red',
        )

    ax.set_ylabel(r'$\Delta E$ [meV]')
    # ax.set_xlabel('')

    fig.savefig('/home/gregor/downloads/acetylacetone.pdf')
    plt.show()


if __name__ == '__main__':
    main()
