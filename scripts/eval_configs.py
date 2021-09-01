import argparse
import logging

import ase.data
import ase.io
import numpy as np
import torch

from e3nnff import data, tools, models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', help='path to XYZ configurations', required=True)
    parser.add_argument('--model', help='path to model', required=True)
    parser.add_argument('--atomic_numbers', help='atomic numbers (comma-separated)', type=str, required=True)
    parser.add_argument('--output', help='output path', required=True)
    parser.add_argument('--r_max', help='distance cutoff (in Ang)', type=float, default=4.0)
    parser.add_argument('--default_dtype',
                        help='set default dtype',
                        type=str,
                        choices=['float32', 'float64'],
                        default='float64')
    return parser.parse_args()


def config_from_atoms(atoms: ase.Atoms) -> data.Configuration:
    atomic_numbers = np.array([ase.data.atomic_numbers[symbol] for symbol in atoms.symbols])
    return data.Configuration(atomic_numbers=atomic_numbers, positions=atoms.positions)


def main():
    args = parse_args()
    tools.set_default_dtype(args.default_dtype)

    atoms_list = ase.io.read(args.configs, format='extxyz', index=':')
    configs = [config_from_atoms(atoms) for atoms in atoms_list]

    zs = [int(z) for z in args.atomic_numbers.split(',')]
    z_table = tools.get_atomic_number_table_from_zs(zs)
    logging.info(z_table)

    loader = data.get_data_loader(
        dataset=[data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max) for config in configs],
        batch_size=len(configs),
        shuffle=False,
        drop_last=False,
    )

    model: models.BodyOrderedModel = torch.load(f=args.model, map_location=torch.device('cpu'))
    output = model.forward(next(iter(loader)), training=False)
    energies = tools.to_numpy(output['energy'])

    # Overwrite info dict
    assert len(energies) == len(atoms_list)
    for atoms, energy in zip(atoms_list, energies):
        atoms.info = {'energy': energy}

    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format='extxyz')


if __name__ == '__main__':
    main()
