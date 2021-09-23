import dataclasses
import logging
import time
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch_geometric
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader

from .checkpoint import CheckpointHandler, CheckpointState
from .torch_tools import to_numpy, tensor_dict_to_device
from .utils import MetricsLogger


@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    device: torch.device,
    swa: Optional[SWAContainer] = None,
):
    lowest_loss = np.inf
    patience_counter = 0

    logging.info('Started training')
    for epoch in range(start_epoch, max_num_epochs):
        # Train
        for batch in train_loader:
            _, opt_metrics = take_step(model=model, loss_fn=loss_fn, batch=batch, optimizer=optimizer, device=device)
            opt_metrics['mode'] = 'opt'
            opt_metrics['epoch'] = epoch
            logger.log(opt_metrics)

        # Validate
        if epoch % eval_interval == 0:
            valid_loss, eval_metrics = evaluate(model=model, loss_fn=loss_fn, data_loader=valid_loader, device=device)
            eval_metrics['mode'] = 'eval'
            eval_metrics['epoch'] = epoch
            logger.log(eval_metrics)

            logging.info(f'Epoch {epoch}: loss={valid_loss:.4f}')

            if valid_loss >= lowest_loss:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f'Stopping optimization after {patience_counter} epochs without improvement')
                    break
            else:
                lowest_loss = valid_loss
                patience_counter = 0
                checkpoint_handler.save(state=CheckpointState(model, optimizer, lr_scheduler), epochs=epoch)

        # LR scheduler and SWA update
        if swa is None or epoch < swa.start:
            lr_scheduler.step()
        else:
            swa.model.update_parameters(model)
            swa.scheduler.step()

    logging.info('Training complete')


def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.data.Batch,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = batch.to(device)
    optimizer.zero_grad()
    output = model(batch, training=True)
    loss = loss_fn(pred=output, ref=batch)
    loss.backward()
    optimizer.step()

    loss_dict = {
        'loss': to_numpy(loss),
        'time': time.time() - start_time,
    }

    return loss, loss_dict


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    total_loss = 0.0
    delta_es_list = []
    delta_fs_list = []

    start_time = time.time()
    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch, training=False)
        batch = batch.cpu()
        output = tensor_dict_to_device(output, device=torch.device('cpu'))

        loss = loss_fn(pred=output, ref=batch)
        total_loss += to_numpy(loss).item()

        delta_es_list.append(batch.energy - output['energy'])
        delta_fs_list.append(batch.forces - output['forces'])

    avg_loss = total_loss / len(data_loader)

    delta_es = torch.cat(delta_es_list, dim=0)
    delta_fs = torch.cat(delta_fs_list, dim=0)

    # MAE: energy and forces
    mae_e = torch.mean(torch.abs(delta_es))
    mae_f = torch.mean(torch.abs(delta_fs))

    # RMSE: energy and forces
    rmse_e = torch.sqrt(torch.mean(torch.square(delta_es)))
    rmse_f = torch.sqrt(torch.mean(torch.square(delta_fs)))

    aux = {
        'loss': avg_loss,
        'mae_e': mae_e.item(),
        'mae_f': mae_f.item(),
        'rmse_e': rmse_e.item(),
        'rmse_f': rmse_f.item(),
        'time': time.time() - start_time,
    }

    return avg_loss, aux
