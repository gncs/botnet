import logging
import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch_geometric
from torch.utils.data import DataLoader

from .torch_tools import to_numpy, tensor_dict_to_device
from .utils import ModelIO, ProgressLogger


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    model_io: ModelIO,
    logger: ProgressLogger,
    eval_interval: int,
    device: torch.device,
):
    lowest_loss = np.inf
    patience_counter = 0
    step = 0

    logging.info('Started training')
    for epoch in range(start_epoch, max_num_epochs):
        for batch in train_loader:
            _, opt_metrics = take_step(model=model, loss_fn=loss_fn, batch=batch, optimizer=optimizer, device=device)
            opt_metrics['mode'] = 'opt'
            opt_metrics['step'] = step
            opt_metrics['epoch'] = epoch
            logger.log(opt_metrics)
            step += 1

        if epoch % eval_interval == 0:
            valid_loss, eval_metrics = evaluate(model=model, loss_fn=loss_fn, data_loader=valid_loader, device=device)
            eval_metrics['mode'] = 'eval'
            eval_metrics['step'] = step
            eval_metrics['epoch'] = epoch
            logger.log(eval_metrics)

            logging.info(f'Epoch {epoch}: {valid_loss:.4f}')

            if valid_loss > lowest_loss:
                patience_counter += 1
                if patience_counter > patience:
                    logging.info(f'Stopping optimization after {patience_counter} epochs without improvement')
                    break
            else:
                lowest_loss = valid_loss
                patience_counter = 0
                model_io.save(model, steps=epoch)

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
    output = model(batch)
    loss = loss_fn(pred=output, ref=batch)
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

    start_time = time.time()
    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch, training=False)
        batch = batch.cpu()
        output = tensor_dict_to_device(output, device=torch.device('cpu'))

        loss = loss_fn(pred=output, ref=batch)
        total_loss += to_numpy(loss).item()

        delta_es.append(torch.abs(batch.energy - output['energy']))

    loss = total_loss / len(data_loader)

    # MAE energy
    delta_e = torch.cat(delta_es)  # [n_graphs, ]
    mae_e = torch.mean(delta_e)

    loss_dict = {
        'loss': loss,
        'mae_e': mae_e.item(),
        'time': time.time() - start_time,
    }

    return loss, loss_dict
