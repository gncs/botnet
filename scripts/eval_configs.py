import argparse

import ase.data
import ase.io
import numpy as np
import torch

from e3nnff import data, tools, modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', help='path to XYZ configurations', required=True)
    parser.add_argument('--model', help='path to model', required=True)
    parser.add_argument('--atomic_numbers', help='atomic numbers (comma-separated)', type=str, required=True)
    parser.add_argument('--output', help='output path', required=True)
    parser.add_argument('--r_max', help='distance cutoff (in Ang)', type=float, default=4.0)
    parser.add_argument('--device', help='select device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--default_dtype',
                        help='set default dtype',
                        type=str,
                        choices=['float32', 'float64'],
                        default='float64')
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    return parser.parse_args()


def config_from_atoms(atoms: ase.Atoms) -> data.Configuration:
    atomic_numbers = np.array([ase.data.atomic_numbers[symbol] for symbol in atoms.symbols])
    return data.Configuration(atomic_numbers=atomic_numbers, positions=atoms.positions)


def main():
    args = parse_args()
    tools.set_default_dtype(args.default_dtype)
    device = tools.init_device(args.device)

    # Load data and prepare input
    atoms_list = ase.io.read(args.configs, format='extxyz', index=':')
    configs = [config_from_atoms(atoms) for atoms in atoms_list]

    z_table = tools.AtomicNumberTable([int(z) for z in args.atomic_numbers.split(',')])

    data_loader = data.get_data_loader(
        dataset=[data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max) for config in configs],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Load model
    model: modules.BodyOrderedModel = torch.load(f=args.model, map_location=device)

    # Collect data
    energies_list = []
    contributions_list = []
    forces_collection = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch, training=False)
        energies_list.append(tools.to_numpy(output['energy']))
        contributions_list.append(tools.to_numpy(output['contributions']))

        forces = np.split(tools.to_numpy(output['forces']), indices_or_sections=batch.ptr[1:], axis=0)
        forces_collection.append(forces[:-1])  # drop last as its emtpy

    energies = np.concatenate(energies_list, axis=0)
    contributions = np.concatenate(contributions_list, axis=0)
    forces_list = [forces for forces_list in forces_collection for forces in forces_list]

    # Store data in atoms objects
    assert len(atoms_list) == len(energies) == contributions.shape[0] == len(forces_list)
    for atoms, energy, contribution, forces in zip(atoms_list, energies, contributions, forces_list):
        atoms.calc = None  # crucial
        atoms.info['energy'] = energy
        atoms.info['contributions'] = contribution
        atoms.arrays['forces'] = forces

    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format='extxyz')


if __name__ == '__main__':
    main()
