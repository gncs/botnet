import logging
import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch_geometric
from torch.utils.data import DataLoader

from .tools import ModelIO, ProgressLogger
from .torch_tools import to_numpy


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

            logging.info(f'Step {epoch}: {valid_loss:.3f}')

            if valid_loss > lowest_loss:
                patience_counter += 1
                if patience_counter > patience:
                    logging.info(f'Stopping optimization after {patience_counter} steps without improvement')
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

    optimizer.zero_grad()
    batch.to(device)
    output = model(batch)
    loss = loss_fn(predictions=output, batch=batch)
    optimizer.step()

    loss_dict = {
        'total_loss': to_numpy(loss),
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

    start_time = time.time()
    for batch in data_loader:
        batch.to(device)
        output = model(batch)
        loss = loss_fn(predictions=output, batch=batch)
        total_loss += to_numpy(loss)

    loss_dict = {
        'total_loss': total_loss,
        'time': time.time() - start_time,
    }

    return total_loss, loss_dict
