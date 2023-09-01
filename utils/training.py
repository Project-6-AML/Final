from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sacred import Experiment
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
#from visdom_logger import VisdomLogger
from copy import deepcopy

from utils.metrics import *

import os
import subprocess as sp

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

###################################################################


def train_global(model: nn.Module,
        loader: DataLoader,
        class_loss: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        epoch: int,
        ex: Experiment = None) -> None:
    
    model.train()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    loader_length = len(loader)
    
    train_losses = AverageMeter(device=device, length=loader_length)
    train_accs = AverageMeter(device=device, length=loader_length)

    pbar = tqdm(loader, ncols=80, desc='Training   [{:03d}]'.format(epoch))
    for i, (batch, labels, indices) in enumerate(pbar):
        batch, labels, indices = map(to_device, (batch, labels, indices))

        ##################################################
        ## extract features
        logits, features, _ = model(batch)
        loss = class_loss(logits, features, labels).mean()
        acc = (logits.detach().argmax(1) == labels).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss)
        train_accs.append(acc)
    
    scheduler.step()

    if ex is not None:
        for i, (loss, acc) in enumerate(zip(train_losses.values_list, train_accs.values_list)):
            step = epoch + i / loader_length
            ex.log_scalar('train.loss', loss, step=step)
            ex.log_scalar('train.acc', acc, step=step)


def train_rerank(model: nn.Module,
        loader: DataLoader,
        class_loss: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        epoch: int,
        ex: Experiment = None) -> None:

    model.train()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    loader_length = len(loader)
    train_losses = AverageMeter(device=device, length=loader_length)
    train_accs = AverageMeter(device=device, length=loader_length)

    pbar = tqdm(loader, ncols=80, desc='Training   [{:03d}]'.format(epoch))
    for i, (batch, labels, indices) in enumerate(pbar):
        batch, labels, indices = map(to_device, (batch, labels, indices))

        ##################################################
        ## extract features
        l = model(batch)
        anchors   = l[0::3]
        positives = l[1::3]
        negatives = l[2::3]
        #print(f"anchors: {anchors.size()}, positives: {positives.size()}, negatives: {negatives.size()}")
        p_logits = model(None, True, src_global=None, src_local=anchors, tgt_global=None, tgt_local=positives)
        n_logits = model(None, True, src_global=None, src_local=anchors, tgt_global=None, tgt_local=negatives)
        logits = torch.cat([p_logits, n_logits], 0)

        bsize = logits.size(0)
        labels = logits.new_ones(logits.size()) * 100
        labels[(bsize//2):] = 0
        loss = class_loss(logits, labels).mean()
        acc = ((torch.sigmoid(logits) > 0.5).long() == labels.long()).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss)
        train_accs.append(acc)

        if not (i + 1) % 100:
            print(f"Loss at batch {i+1}: {loss.item()}")
            print(f"Logits: {logits}")

    scheduler.step()

    if ex is not None:
        for i, (loss, acc) in enumerate(zip(train_losses.values_list, train_accs.values_list)):
            step = epoch + i / loader_length
            ex.log_scalar('train.loss', loss, step=step)
            ex.log_scalar('train.acc', acc, step=step)

    return loss

def train_rerank_backbone(model: nn.Module,
        loader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        epoch: int,
        ex: Experiment = None) -> None:
    
    model.train()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    loader_length = len(loader)
    train_losses = AverageMeter(device=device, length=loader_length)
    train_accs = AverageMeter(device=device, length=loader_length)

    class_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    save_size = 10
    save_order = 0

    pbar = tqdm(loader, ncols=80, desc='Training   [{:03d}]'.format(epoch))
    for i, (batch, labels, indices) in enumerate(pbar):
        batch, labels, indices = map(to_device, (batch, labels, indices))

        ##################################################
        ## extract features
        l = model(batch)
        anchors   = l[0::3]
        positives = l[1::3]
        negatives = l[2::3]

        loss = class_loss(anchors, positives, negatives)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not (i + 1) % 100:
            print(f"Loss at batch {i+1}: {loss.item()}")

    scheduler.step()

def train_rerank_transformer(model: nn.Module,
        loader: DataLoader,
        class_loss: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        epoch: int,
        ex: Experiment = None) -> None:
    
    model.train()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    loader_length = len(loader)
    train_losses = AverageMeter(device=device, length=loader_length)
    train_accs = AverageMeter(device=device, length=loader_length)
    
    offset = 0
    flag = 0
    save_order = 0
    save_size = 10
    features = "Prova"

    pbar = tqdm(loader, ncols=80, desc='Training   [{:03d}]'.format(epoch))
    for i, (batch, labels, indices) in enumerate(pbar):
        batch, labels, indices = map(to_device, (batch, labels, indices))

        if flag % save_size == 0:
            del features        
            offset=0
            features = torch.load(f"/home/cristiano/Desktop/Features/features_{save_order}.pt") # Carico save_size batches
            features = to_device(features)
            save_order += 1

        ##################################################
        extracted = features[offset : offset + batch.size(0)]
        offset += batch.size(0)

        ## extract features
        anchors   = extracted[0::3]
        positives = extracted[1::3]
        negatives = extracted[2::3]
        #print(f"anchors: {anchors.size()}, positives: {positives.size()}, negatives: {negatives.size()}")
        p_logits = model(src_global=None, src_local=anchors, tgt_global=None, tgt_local=positives)
        n_logits = model(src_global=None, src_local=anchors, tgt_global=None, tgt_local=negatives)
        logits = torch.cat([p_logits, n_logits], 0)

        bsize = logits.size(0)
        labels = logits.new_ones(logits.size())
        labels[(bsize//2):] = 0
        loss = class_loss(logits, None, labels).mean()
        acc = ((torch.sigmoid(logits) > 0.5).long() == labels.long()).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss)
        train_accs.append(acc)

        flag += 1

        if not (i + 1) % 100:
            print(f"Loss at batch {i+1}: {loss.item()}")
            print(f"Logits: {logits}")

    scheduler.step()



###################################################################


def evaluate_global(model: nn.Module,
        query_loader: DataLoader,
        gallery_loader: Optional[DataLoader],
        recall_ks: List[int]):
    
    model.eval()
    device = next(model.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    all_query_features, all_query_labels = [], []
    all_gallery_features, all_gallery_labels = None, None

    with torch.no_grad():
        for batch, labels, _ in tqdm(query_loader, desc='Extracting query features', leave=False, ncols=80):
            batch, labels = map(to_device, (batch, labels))
            features = model(batch)[1]
            all_query_labels.append(labels)
            all_query_features.append(features)
        all_query_labels = torch.cat(all_query_labels, 0)
        all_query_features = torch.cat(all_query_features, 0)

        if gallery_loader is not None:
            all_gallery_features, all_gallery_labels = [], []
            for batch, labels, _ in tqdm(gallery_loader, desc='Extracting gallery features', leave=False, ncols=80):
                batch, labels = map(to_device, (batch, labels))
                features = model(batch)[1]
                all_gallery_labels.append(labels)
                all_gallery_features.append(features)
            all_gallery_labels = torch.cat(all_gallery_labels, 0)
            all_gallery_features = torch.cat(all_gallery_features, 0)

        recall_function = partial(
            recall_at_ks, query_features=all_query_features, query_labels=all_query_labels, ks=recall_ks,
            gallery_features=all_gallery_features, gallery_labels=all_gallery_labels
        )
        recalls_cosine, nn_dists, nn_inds = recall_function(cosine=True)

        del all_query_features, all_query_labels, all_gallery_features, all_gallery_labels
        torch.cuda.empty_cache()
    
    return recalls_cosine, nn_dists, nn_inds


def evaluate_rerank(backbone: nn.Module,
            transformer: nn.Module,
            cache_nn_inds: torch.Tensor,
            query_loader: DataLoader,
            gallery_loader: Optional[DataLoader],
            recall_ks: List[int]):

    backbone.eval()
    transformer.eval()
    device = next(backbone.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    all_query_features, all_query_labels = [], []
    all_gallery_features, all_gallery_labels = None, None

    with torch.no_grad():
        for batch, labels, _ in tqdm(query_loader, desc='Extracting query features', leave=False, ncols=80):
            batch, labels = map(to_device, (batch, labels))
            features = backbone(batch)
            all_query_labels.append(labels)
            all_query_features.append(features.cpu())
        all_query_features = torch.cat(all_query_features, 0)
        all_query_labels = torch.cat(all_query_labels, 0)

        if gallery_loader is not None:
            all_gallery_features = []
            all_gallery_labels = []
            for batch, labels, _ in tqdm(gallery_loader, desc='Extracting gallery features', leave=False, ncols=80):
                batch, labels = map(to_device, (batch, labels))
                features = backbone(batch)
                all_gallery_labels.append(labels.cpu())
                all_gallery_features.append(features.cpu())

            all_gallery_labels = torch.cat(all_gallery_labels, 0)
            all_gallery_features = torch.cat(all_gallery_features, 0)

        recall_function = partial(
                recall_at_ks_rerank, 
                query_features=all_query_features.cpu(), query_labels=all_query_labels.cpu(), ks=recall_ks,
                matcher=transformer, cache_nn_inds=cache_nn_inds,
                gallery_features=all_gallery_features, gallery_labels=all_gallery_labels
            )
        recalls_rerank, nn_dists, nn_inds = recall_function()

    print(f"Free GPU memory before deleting: {get_gpu_memory()}")
    del all_query_features, all_query_labels, all_gallery_features, all_gallery_labels
    torch.cuda.empty_cache()
    print(f"Free GPU memory after deleting: {get_gpu_memory()}")

    return recalls_rerank, nn_dists, nn_inds


def evaluate_rerank_all(model: nn.Module,
            cache_nn_inds: torch.Tensor,
            query_loader: DataLoader,
            gallery_loader: Optional[DataLoader],
            recall_ks: List[int]):

    print(f"Free GPU memory before: {get_gpu_memory()}")
    model.eval()
    device = next(model.parameters()).device
    print(f"Device in evaluation: {device}")
    to_device = lambda x: x.to(device, non_blocking=True)
    all_query_features, all_query_labels = [], []
    all_gallery_features, all_gallery_labels = None, None

    with torch.no_grad():
        for batch, labels, _ in tqdm(query_loader, desc='Extracting query features', leave=False, ncols=80):
            batch, labels = map(to_device, (batch, labels))
            features = model(batch)
            all_query_labels.append(labels.cpu())
            all_query_features.append(features.cpu())
        all_query_features = torch.cat(all_query_features, 0)
        all_query_labels = torch.cat(all_query_labels, 0)

        print(f"all_query_features size: {all_query_features.size()}")

        if gallery_loader is not None:
            all_gallery_features = []
            all_gallery_labels = []
            for batch, labels, _ in tqdm(gallery_loader, desc='Extracting gallery features', leave=False, ncols=80):
                batch, labels = map(to_device, (batch, labels))
                features = model(batch)
                all_gallery_labels.append(labels.cpu())
                all_gallery_features.append(features.cpu())

            all_gallery_labels = torch.cat(all_gallery_labels, 0)
            all_gallery_features = torch.cat(all_gallery_features, 0)

        recall_function = partial(
                recall_at_ks_rerank, 
                query_features=all_query_features.cpu(), query_labels=all_query_labels.cpu(), ks=recall_ks,
                model=model, cache_nn_inds=cache_nn_inds,
                gallery_features=all_gallery_features, gallery_labels=all_gallery_labels
            )
        recalls_rerank, nn_dists, nn_inds = recall_function()

    #print(f"Free GPU memory before deleting: {get_gpu_memory()}")
    del all_query_features, all_query_labels, all_gallery_features, all_gallery_labels
    torch.cuda.empty_cache()
    #print(f"Free GPU memory after deleting: {get_gpu_memory()}")

    return recalls_rerank, nn_dists, nn_inds
