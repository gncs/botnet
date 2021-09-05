import logging
import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch_geometric
from torch.utils.data import DataLoader

from .checkpoint import CheckpointHandler, CheckpointState
from .torch_tools import to_numpy, tensor_dict_to_device
from .utils import ProgressLogger


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
    logger: ProgressLogger,
    eval_interval: int,
    device: torch.device,
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

        # LR scheduler
        lr_scheduler.step()

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
    delta_es = []
    delta_fs = []

    start_time = time.time()
    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch, training=False)
        batch = batch.cpu()
        output = tensor_dict_to_device(output, device=torch.device('cpu'))

        loss = loss_fn(pred=output, ref=batch)
        total_loss += to_numpy(loss).item()

        delta_es.append(torch.abs(batch.energy - output['energy']))
        delta_fs.append(torch.abs(batch.forces - output['forces']))

    avg_loss = total_loss / len(data_loader)

    # MAE energy and forces
    mae_e = torch.mean(torch.cat(delta_es, dim=0))
    mae_f = torch.mean(torch.cat(delta_fs, dim=0))

    aux = {
        'loss': avg_loss,
        'mae_e': mae_e.item(),
        'mae_f': mae_f.item(),
        'time': time.time() - start_time,
    }

    return avg_loss, aux
