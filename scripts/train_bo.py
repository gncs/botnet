import logging
import os

import numpy as np
import torch.nn
from e3nn import o3

from e3nnff import data, tools, models, modules


def main() -> None:
    parser = tools.build_default_arg_parser()
    parser = tools.add_rmd17_parser(parser)
    args = parser.parse_args()

    tag = tools.get_tag(name=args.name, seed=args.seed)

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
    logging.info(f'Configuration: {args}')
    device = tools.init_device(args.device)
    tools.set_default_dtype(args.dtype)

    # Data preparation
    train_valid_configs, test_configs = data.load_rmd17(
        directory=args.downloads_dir,
        subset=args.subset,
        split=args.split,
    )
    train_valid_configs = train_valid_configs[:args.max_size_train]
    test_configs = test_configs[:args.max_size_test]

    train_configs, valid_configs = data.split_train_valid_configs(train_valid_configs,
                                                                  valid_fraction=args.valid_fraction)
    logging.info(f'Number of configurations: train={len(train_configs)}, valid={len(valid_configs)}, '
                 f'test={len(test_configs)}')

    # Atomic number table
    # yapf: disable
    z_table = tools.get_atomic_number_table_from_zs(
        z
        for configs in (train_configs, valid_configs)
        for config in configs
        for z in config.atomic_numbers
    )
    # yapf: enable
    logging.info(z_table)
    atomic_energies = np.array([data.rmd17_atomic_energies[z] for z in z_table.zs])

    # yapf: disable
    train_loader, valid_loader, test_loader = (
        data.get_data_loader(
            dataset=[data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max) for config in configs],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        for configs in (train_configs, valid_configs, test_configs)
    )
    # yapf: enable

    loss_fn: torch.nn.Module
    if args.no_forces:
        loss_fn = modules.EnergyLoss()
    else:
        loss_fn = modules.EnergyForcesLoss(energy_weight=1.0, forces_weight=100.0)
    logging.info(loss_fn)

    # Build model
    logging.info('Building model')
    model = models.BodyOrderedModel(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        include_forces=True,
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
        start_epoch = checkpoint_handler.load_latest(state=tools.CheckpointState(model, optimizer, lr_scheduler))

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

    # Evaluation on test dataset
    epoch = checkpoint_handler.load_latest(state=tools.CheckpointState(model, optimizer, lr_scheduler))
    test_loss, test_metrics = tools.evaluate(model, loss_fn=loss_fn, data_loader=test_loader, device=device)
    test_metrics['mode'] = 'test'
    test_metrics['epoch'] = epoch
    logger.log(test_metrics)
    logging.info(f'Test loss (epoch {epoch}): {test_loss:.3f}')

    # Save entire model
    model_path = os.path.join(args.checkpoints_dir, tag + '.model')
    logging.info(f'Saving model to {model_path}')
    torch.save(model, model_path)

    logging.info('Done')


if __name__ == '__main__':
    main()
