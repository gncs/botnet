import argparse
import dataclasses
import logging
import os
from typing import Optional, Sequence, Tuple, Dict

import numpy as np
import torch.nn
from e3nn import o3

from e3nnff import data, tools, modules


def add_rmd17_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--dataset',
                        help='dataset name',
                        type=str,
                        choices=['iso17', 'rmd17', '3bpa', 'acac'],
                        required=True)
    parser.add_argument('--subset', help='subset name')
    parser.add_argument('--split', help='train test split', type=int)
    return parser


@dataclasses.dataclass
class DatasetCollection:
    train: data.Configurations
    valid: data.Configurations
    tests: Sequence[Tuple[str, data.Configurations]]


def get_dataset(downloads_dir: str, dataset: str, subset: Optional[str], split: Optional[int]) -> DatasetCollection:
    if dataset == 'iso17':
        logging.info(f'Dataset: {dataset}')
        ref_configs, test_within, test_other = data.load_iso17(directory=downloads_dir)
        train_size, valid_size = 5000, 500
        train_valid_configs = np.random.choice(ref_configs, train_size + valid_size)
        train_configs, valid_configs = train_valid_configs[:train_size], train_valid_configs[train_size:]
        return DatasetCollection(train=train_configs,
                                 valid=valid_configs,
                                 tests=[('test_within', test_within), ('test_other', test_other)])

    if dataset == 'rmd17':
        if not subset or not split:
            raise RuntimeError('Specify subset and split')
        logging.info(f'Dataset: {dataset}, subset: {subset}')
        train_valid_configs, test_configs = data.load_rmd17(directory=downloads_dir, subset=subset, split=split)
        train_configs, valid_configs = data.split_train_valid_configs(configs=train_valid_configs, valid_fraction=0.1)
        return DatasetCollection(train=train_configs, valid=valid_configs, tests=[('test', test_configs)])

    if dataset == '3bpa':
        if not subset:
            raise RuntimeError('Specify subset')
        logging.info(f'Dataset: {dataset}, training: {subset}')
        configs_dict = data.load_3bpa(directory=downloads_dir)
        train_valid_configs = configs_dict[subset]
        train_configs, valid_configs = data.split_train_valid_configs(configs=train_valid_configs, valid_fraction=0.1)
        return DatasetCollection(train=train_configs,
                                 valid=valid_configs,
                                 tests=[(key, configs_dict[key]) for key in ['test300K', 'test600K', 'test1200K']])

    if dataset == 'acac':
        if not subset:
            raise RuntimeError('Specify subset')
        logging.info(f'Dataset: {dataset}, training: {subset}')
        configs_dict = data.load_acac(directory=downloads_dir)
        train_valid_configs = configs_dict[subset]
        train_configs, valid_configs = data.split_train_valid_configs(configs=train_valid_configs, valid_fraction=0.1)
        return DatasetCollection(train=train_configs,
                                 valid=valid_configs,
                                 tests=[(key, configs_dict[key]) for key in ['test_MD_300K', 'test_MD_600K']])

    raise RuntimeError(f'Unknown dataset: {dataset}')


atomic_energies_dict: Dict[str, Dict[int, float]] = {
    'iso17': data.iso17_atomic_energies,
    'rmd17': data.rmd17_atomic_energies,
    '3bpa': data.three_bpa_atomic_energies,
    'acac': data.acac_atomic_energies,
}


def main() -> None:
    parser = tools.build_default_arg_parser()
    parser = add_rmd17_parser(parser)
    args = parser.parse_args()

    tag = tools.get_tag(name=args.name, seed=args.seed)

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
    logging.info(f'Configuration: {args}')
    device = tools.init_device(args.device)
    tools.set_default_dtype(args.default_dtype)

    # Data preparation
    collections = get_dataset(downloads_dir=args.downloads_dir,
                              dataset=args.dataset,
                              subset=args.subset,
                              split=args.split)
    logging.info(f'Number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, '
                 f'tests={[len(test_configs) for name, test_configs in collections.tests]}')

    # Atomic number table
    # yapf: disable
    z_table = tools.get_atomic_number_table_from_zs(
        z
        for configs in (collections.train, collections.valid)
        for config in configs
        for z in config.atomic_numbers
    )
    # yapf: enable
    logging.info(z_table)
    atomic_energies = np.array([atomic_energies_dict[args.dataset][z] for z in z_table.zs])

    # yapf: disable
    train_loader, valid_loader = (
        data.get_data_loader(
            dataset=[data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max) for config in configs],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        for configs in (collections.train, collections.valid)
    )
    # yapf: enable

    loss_fn = modules.EnergyForcesLoss(energy_weight=1.0, forces_weight=100.0)
    logging.info(loss_fn)

    # Build model
    logging.info('Building model')
    model = modules.BodyOrderedModel(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
    )
    model.to(device)
    logging.info(model)
    logging.info(f'Number of parameters: {tools.count_parameters(model)}')

    optimizer = tools.get_optimizer(name=args.optimizer, learning_rate=args.lr, parameters=model.parameters())
    logger = tools.ProgressLogger(directory=args.results_dir, tag=tag)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.lr_scheduler_gamma)

    checkpoint_handler = tools.CheckpointHandler(directory=args.checkpoints_dir, tag=tag, keep=args.keep_checkpoints)

    start_epoch = 0
    if args.restart_latest:
        start_epoch = checkpoint_handler.load_latest(state=tools.CheckpointState(model, optimizer, lr_scheduler),
                                                     device=device)

    logging.info(f'Optimizer: {optimizer}')

    tools.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        device=device,
    )

    # Evaluation on test datasets
    epoch = checkpoint_handler.load_latest(state=tools.CheckpointState(model, optimizer, lr_scheduler), device=device)
    logging.info(f'Loading model from epoch {epoch}')

    logging.info('Running tests')
    for name, test_set in collections.tests:
        test_loader = data.get_data_loader(
            dataset=[data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max) for config in test_set],
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        test_loss, test_metrics = tools.evaluate(model, loss_fn=loss_fn, data_loader=test_loader, device=device)
        logging.info(f"Test set '{name}': "
                     f'loss={test_loss:.3f}, '
                     f'mae_e={test_metrics["mae_e"] * 1000:.3f} meV, '
                     f'mae_f={test_metrics["mae_f"] * 1000:.3f} meV/Ang')

    # Save entire model
    model_path = os.path.join(args.checkpoints_dir, tag + '.model')
    logging.info(f'Saving model to {model_path}')
    torch.save(model, model_path)

    logging.info('Done')


if __name__ == '__main__':
    main()
