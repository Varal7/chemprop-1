import logging
from typing import Callable, Dict

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import numpy as np
from collections import Counter

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(model: nn.Module,
          data_loader: MoleculeDataLoader,
          additional_dataloaders: Dict[str, MoleculeDataLoader],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None,
          context = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data_loader: A MoleculeDataLoader.
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    model.train()
    loss_sum, iter_count = 0, 0
    main_loss_sum, distill_loss_sum = 0, 0

    additional_losses_sum = Counter()

    iterable_dataloaders = {"main": iter(data_loader)}

    iterable_dataloaders.update({k: iter(d) for k, d in additional_dataloaders.items()})


    for _ in tqdm(range(len(data_loader))):
        # Prepare batch
        for name in iterable_dataloaders.keys():
            try:
                batch : MoleculeDataset = next(iterable_dataloaders[name])
            except StopIteration:
                assert name != "main"
                iterable_dataloaders[name] = iter(additional_dataloaders[name])

            mol_batch, features_batch, target_batch, target_features_batch = batch.batch_graph(), batch.features(), batch.targets(), batch.target_features()
            if 'images' in context:
                target_features_batch = context['images'].get_item(batch.smiles())
            mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
            targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

            # Run model
            model.zero_grad()
            preds, local_context = model(mol_batch, features_batch)

            # Move tensors to correct device
            mask = mask.to(preds.device)
            targets = targets.to(preds.device)
            class_weights = torch.ones(targets.shape, device=preds.device)

            def compute_loss(preds):
                if args.dataset_type == 'multiclass':
                    targets_l = targets.long()
                    main_loss = torch.cat([loss_func(preds[:, target_index, :], targets_l[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
                else:
                    main_loss = loss_func(preds, targets) * class_weights * mask

                main_loss = main_loss.sum() / mask.sum()

                return main_loss

            local_context['loss'] = compute_loss(preds)
            local_context['compute_loss_fn'] = compute_loss

            context['model.ffn'] = model.ffn

            context['device'] = preds.device

            if target_features_batch is not None:
                if not 'images' in context:
                    initial_size = target_features_batch.size()
                    target_features_mask = torch.Tensor([[x is not None for x in tb] for tb in target_features_batch.view(initial_size[0], -1)]).view(initial_size)
                    target_features_batch = torch.Tensor([[0 if x != x else x for x in tb] for tb in target_features_batch.view(initial_size[0], -1)]).view(initial_size)
                else:
                    target_features_mask = torch.ones_like(target_features_batch)

                local_context['target_features_mask'] = target_features_mask.to(preds.device)
                local_context['target_features_batch'] = target_features_batch.to(preds.device)


                if model.use_distill and not model.distill.accepts_multi_images:
                    local_context['target_features_mask'] = local_context['target_features_mask'].squeeze(1)
                    local_context['target_features_batch'] = local_context['target_features_batch'].squeeze(1)


            key_prefix = "" if name == "main" else f"{name}_"
            context.update({(key_prefix + k): v for k, v in local_context.items()})



        if model.use_distill:
            distill_loss = model.distill.compute_loss(context)
            additional_losses_to_log = model.distill.additional_losses_to_log()
        else:
            distill_loss = torch.tensor(0).to(context['device'])

        loss = args.main_loss_lambda * context['loss'] + distill_loss

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        iter_count += len(batch)

        if model.use_distill:
            main_loss_sum += context['loss'].sum()
            distill_loss_sum += distill_loss.sum()
            for key, value in additional_losses_to_log.items():
                additional_losses_sum[key] += value
                context[key] = value

            model.distill.update_meters(context)


        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            main_loss_avg = main_loss_sum / iter_count
            distill_loss_avg = distill_loss_sum / iter_count


            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            if model.use_distill:
                distill_string = f'Main loss = {main_loss_avg:.4f}, scaled distill loss = {distill_loss_avg:.4f}, '
                for key, val in additional_losses_sum.items():
                    distill_string += f'{key} = {val / iter_count:.4f}'
                    additional_losses_sum[key] = 0
            else:
                distill_string = ''

            loss_sum, iter_count = 0, 0
            distill_loss_sum = 0
            main_loss_sum = 0

            debug(f'Loss = {loss_avg:.4e}, {distill_string}PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                if model.use_distill:
                    writer.add_scalar('main_loss', main_loss_avg, n_iter)
                    writer.add_scalar('scaled_distill_loss', distill_loss_avg, n_iter)

                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter
