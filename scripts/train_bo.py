import logging

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

    # Data preparation
    train_valid_test_configs = data.load_rmd17(
        directory=args.downloads_dir,
        subset=args.subset,
        valid_fraction=args.valid_fraction,
        split=args.split,
        max_size_train=args.max_size_train,
        max_size_test=args.max_size_test,
    )

    # Atomic number table
    # yapf: disable
    z_table = tools.get_atomic_number_table_from_zs(
        z
        for configs in train_valid_test_configs[:2]  # train and valid configs
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
        for configs in train_valid_test_configs
    )
    # yapf: enable

    include_forces = True
    loss_fn: torch.nn.Module
    if include_forces:
        loss_fn = modules.EnergyForcesLoss(energy_weight=1.0, forces_weight=100.0)
    else:
        loss_fn = modules.EnergyLoss()

    mean_atom_inter, std_atom_inter = modules.compute_mean_std_atomic_inter_energy(train_loader, atomic_energies)

    # Build model
    logging.info('Building model')
    model = models.SimpleBodyOrderedModel(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        atomic_inter_scale=std_atom_inter,
        atomic_inter_shift=mean_atom_inter,
        include_forces=include_forces,
    )
    model.to(device)
    logging.info(f'Number of parameters in {model.__class__.__name__}: {tools.count_parameters(model)}')

    optimizer = tools.get_optimizer(name=args.optimizer, learning_rate=args.lr, parameters=model.parameters())
    logger = tools.ProgressLogger(directory=args.results_dir, tag=tag)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.lr_scheduler_gamma)

    io = tools.CheckpointIO(directory=args.checkpoints_dir, tag=tag, keep=args.keep_models)
    builder = tools.CheckpointBuilder(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    handler = tools.CheckpointHandler(builder, io)

    tools.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=handler,
        eval_interval=args.eval_interval,
        start_epoch=0,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        device=device,
    )

    # Evaluation on test dataset
    handler.load_latest()
    test_loss, test_metrics = tools.evaluate(model, loss_fn=loss_fn, data_loader=test_loader, device=device)
    test_metrics['mode'] = 'test'
    logger.log(test_metrics)
    logging.info(f'Test loss: {test_loss:.3f}')

    logging.info('Done')


if __name__ == '__main__':
    main()
