import argparse
import dataclasses
import logging
import os
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch.nn
from e3nn import o3
from torch.optim.swa_utils import AveragedModel, SWALR

from e3nnff import data, tools, modules


def add_train_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--dataset',
                        help='dataset name',
                        type=str,
                        choices=['iso17', 'rmd17', '3bpa', 'acac', 'ethanol'],
                        required=True)
    parser.add_argument('--subset', help='subset name')
    parser.add_argument('--split', help='train test split', type=int)
    parser.add_argument('--model',
                        help='model type',
                        default='body_ordered',
                        choices=['body_ordered', 'scale_shift', 'single_readout'])
    parser.add_argument('--loss', help='type of loss', default='default', choices=['default', 'ace'])
    return parser


@dataclasses.dataclass
class SubsetCollection:
    train: data.Configurations
    valid: data.Configurations
    tests: List[Tuple[str, data.Configurations]]


def get_dataset(downloads_dir: str, dataset: str, subset: Optional[str], split: Optional[int]) -> SubsetCollection:
    if dataset == 'iso17':
        logging.info(f'Dataset: {dataset}')
        ref_configs, test_within, test_other = data.load_iso17(directory=downloads_dir)
        train_size, valid_size = 5000, 500
        train_valid_configs = np.random.choice(ref_configs, train_size + valid_size)
        train_configs, valid_configs = train_valid_configs[:train_size], train_valid_configs[train_size:]
        return SubsetCollection(train=train_configs,
                                valid=valid_configs,
                                tests=[('test_within', test_within), ('test_other', test_other)])

    if dataset == 'rmd17':
        if not subset or not split:
            raise RuntimeError('Specify subset and split')
        logging.info(f'Dataset: {dataset}, subset: {subset}')
        train_valid_configs, test_configs = data.load_rmd17(directory=downloads_dir, subset=subset, split=split)
        train_configs, valid_configs = data.random_train_valid_split(items=train_valid_configs, valid_fraction=0.05)
        return SubsetCollection(train=train_configs, valid=valid_configs, tests=[('test', test_configs)])

    if dataset == '3bpa':
        if not subset:
            raise RuntimeError('Specify subset')
        logging.info(f'Dataset: {dataset}, training: {subset}')
        configs_dict = data.load_3bpa(directory=downloads_dir)
        train_valid_configs = configs_dict[subset]
        train_configs, valid_configs = data.random_train_valid_split(items=train_valid_configs, valid_fraction=0.10)
        return SubsetCollection(train=train_configs,
                                valid=valid_configs,
                                tests=[(key, configs_dict[key]) for key in ['test_300K', 'test_600K', 'test_1200K']])

    if dataset == 'acac':
        if not subset:
            raise RuntimeError('Specify subset')
        logging.info(f'Dataset: {dataset}, training: {subset}')
        configs_dict = data.load_acac(directory=downloads_dir)
        train_valid_configs = configs_dict[subset]
        train_configs, valid_configs = data.random_train_valid_split(items=train_valid_configs, valid_fraction=0.10)
        return SubsetCollection(train=train_configs,
                                valid=valid_configs,
                                tests=[(key, configs_dict[key]) for key in ['test_MD_300K', 'test_MD_600K']])

    if dataset == 'ethanol':
        logging.info(f'Dataset: {dataset}')
        configs_dict = data.load_ethanol(directory=downloads_dir)
        train_valid_configs = configs_dict['train']
        train_configs, valid_configs = data.random_train_valid_split(items=train_valid_configs, valid_fraction=0.05)
        return SubsetCollection(train=train_configs, valid=valid_configs, tests=[('test_MD', configs_dict['test_MD'])])

    raise RuntimeError(f'Unknown dataset: {dataset}')


atomic_energies_dict: Dict[str, Dict[int, float]] = {
    'iso17': data.iso17_atomic_energies,
    'rmd17': data.rmd17_atomic_energies,
    '3bpa': data.three_bpa_atomic_energies,
    'acac': data.acac_atomic_energies,
    'ethanol': data.ethanol_atomic_energies,
}


def main() -> None:
    parser = tools.build_default_arg_parser()
    parser = add_train_parser(parser)
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

    train_loader = data.get_data_loader(
        dataset=[data.AtomicData.from_config(c, z_table=z_table, cutoff=args.r_max) for c in collections.train],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = data.get_data_loader(
        dataset=[data.AtomicData.from_config(c, z_table=z_table, cutoff=args.r_max) for c in collections.valid],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    loss_fn: torch.nn.Module
    if args.loss == 'ace':
        loss_fn = modules.ACELoss(energy_weight=15.0, forces_weight=1.0)
    else:
        loss_fn = modules.EnergyForcesLoss(energy_weight=1.0, forces_weight=100.0)
    logging.info(loss_fn)

    # Build model
    logging.info('Building model')
    model_config = dict(
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

    model: torch.nn.Module
    if args.model == 'scale_shift':
        mean, std = modules.compute_mean_std_atomic_inter_energy(train_loader, atomic_energies)
        model = modules.ScaleShiftBodyOrderedModel(
            **model_config,
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
        )
    elif args.model == 'single_readout':
        model = modules.SingleReadoutModel(**model_config)
    else:
        model = modules.BodyOrderedModel(**model_config)

    model.to(device)

    # Optimizer
    param_options = dict(
        params=[{
            'name': 'interactions',
            'params': model.interactions.parameters(),
            'weight_decay': args.weight_decay,
        }, {
            'name': 'readouts',
            'params': model.readouts.parameters(),
            'weight_decay': 0.0,
        }],
        lr=args.lr,
        amsgrad=args.amsgrad,
    )

    optimizer: torch.optim.Optimizer
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(**param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)

    logger = tools.MetricsLogger(directory=args.results_dir, tag=tag + '_train')
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.lr_scheduler_gamma)

    checkpoint_handler = tools.CheckpointHandler(directory=args.checkpoints_dir, tag=tag, keep=args.keep_checkpoints)

    start_epoch = 0
    if args.restart_latest:
        start_epoch = checkpoint_handler.load_latest(state=tools.CheckpointState(model, optimizer, lr_scheduler),
                                                     device=device)

    swa: Optional[tools.SWAContainer] = None
    if args.swa:
        swa = tools.SWAContainer(
            model=AveragedModel(model),
            scheduler=SWALR(optimizer=optimizer, swa_lr=args.lr, anneal_epochs=1, anneal_strategy='linear'),
            start=10,
        )
        logging.info(f'Using stochastic weight averaging (after {swa.start} epochs)')

    logging.info(model)
    logging.info(f'Number of parameters: {tools.count_parameters(model)}')
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
        swa=swa,
    )

    if swa:
        logging.info('Building averaged model')
        # Update batch norm statistics for the swa_model at the end (actually we are not using bn)
        torch.optim.swa_utils.update_bn(train_loader, swa.model)
        model = swa.model.module
    else:
        epoch = checkpoint_handler.load_latest(state=tools.CheckpointState(model, optimizer, lr_scheduler),
                                               device=device)
        logging.info(f'Loaded model from epoch {epoch}')

    # Evaluation on test datasets
    logging.info('Computing metrics for training, validation, and test sets')
    logger = tools.MetricsLogger(directory=args.results_dir, tag=tag + '_eval')
    for name, subset in [('train', collections.train), ('valid', collections.valid)] + collections.tests:
        data_loader = data.get_data_loader(
            dataset=[data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max) for config in subset],
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        loss, metrics = tools.evaluate(model, loss_fn=loss_fn, data_loader=data_loader, device=device)
        logging.info(f"Subset '{name}': "
                     f'loss={loss:.3f}, '
                     f'mae_e={metrics["mae_e"] * 1000:.3f} meV, '
                     f'mae_f={metrics["mae_f"] * 1000:.3f} meV/Ang, '
                     f'rmse_e={metrics["rmse_e"] * 1000:.3f} meV, '
                     f'rmse_f={metrics["rmse_f"] * 1000:.3f} meV/Ang')
        metrics['subset'] = name
        metrics['name'] = args.name
        metrics['seed'] = args.seed
        logger.log(metrics)

    # Save entire model
    model_path = os.path.join(args.checkpoints_dir, tag + '.model')
    logging.info(f'Saving model to {model_path}')
    torch.save(model, model_path)

    logging.info('Done')


if __name__ == '__main__':
    main()
