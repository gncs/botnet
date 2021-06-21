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

    # Build model
    logging.info('Building model')
    model = models.BodyOrderModel(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        num_interactions=args.num_interactions,
        num_channels_input=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
    )
    model.to(device)
    logging.info(f'Number of model parameters: {tools.count_parameters(model)}')

    model_io = tools.ModelIO(directory=args.models_dir, tag=tag, keep=args.keep_models)
    optimizer = tools.get_optimizer(name=args.optimizer,
                                    learning_rate=args.learning_rate,
                                    parameters=model.parameters())
    logger = tools.ProgressLogger(directory=args.results_dir, tag=tag)

    loss_fn = modules.EnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)

    tools.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        model_io=model_io,
        eval_interval=args.eval_interval,
        start_epoch=0,
        max_num_epochs=args.max_num_steps,
        logger=logger,
        patience=args.patience,
        device=device,
    )

    # Evaluation on test dataset
    loaded_model, step = model_io.load_latest(device)
    test_loss, test_metrics = tools.evaluate(loaded_model, loss_fn=loss_fn, data_loader=test_loader, device=device)
    logger.log(test_metrics)
    logging.info(f'Test loss: {test_loss:.3f}')

    logging.info('Done')


if __name__ == '__main__':
    main()
