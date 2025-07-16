import os
import argparse
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import time
import torch.utils
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from data import data_loaders
from model import RandLANet
from utils.tools import SemanticKITTIConfig as cfg
from utils.metrics import get_accuracy_tensor, get_iou_tensor
from utils.utils import load_yaml, compute_avg_grad_norm, get_learning_rate, viz_pointcloud_with_labels

from common.dataset.kitti.kittiDataset import SemanticKITTIDataset

def init_distributed_mode(args):
    if'RANK'in os.environ and'WORLD_SIZE'in os.environ:  
        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = torch.device(f"cuda:{args.local_rank}")
    else:
        print('Not using distributed mode')
        args.distributed = False
    return

def train_one_epoch(model,
                    optimizer,
                    data_loader,
                    criterion,
                    writer,
                    device,
                    num_classes,
                    epoch):
    model.train()
    loss_list = []
    acc_sum_tensor = torch.zeros((2, num_classes), dtype=torch.float64, device=device)
    iou_sum_tensor = torch.zeros((2, num_classes), dtype=torch.float64, device=device)
    for step, input_dict in enumerate(tqdm(data_loader, desc='Training', leave=False)):
        points = input_dict["points"].to(device)
        labels = input_dict["labels"].to(device)
        optimizer.zero_grad()
    
        logp = model(points)
        scores = F.softmax(logp, dim=-1) 
        loss = criterion(logp, labels)
        loss.backward()
        optimizer.step()
        
        mean_loss = reduce_value(loss, average=True)            
        acc_tensor = reduce_value(get_accuracy_tensor(scores, labels, device=device), average=False)
        iou_tensor = reduce_value(get_iou_tensor(scores, labels, device=device), average=False)
        loss_list.append(mean_loss.cpu())
        acc_sum_tensor += acc_tensor
        iou_sum_tensor += iou_tensor

        if writer != None and step % 50 == 0:
            global_step = epoch * len(data_loader) + step
            avg_grad_norm = compute_avg_grad_norm(model)
            current_lr = get_learning_rate(optimizer)
            writer.add_scalar("Train/Loss", mean_loss, global_step)   
            writer.add_scalar("Train/GradNorm", avg_grad_norm, global_step)
            writer.add_scalar("Train/LearningRate", current_lr, global_step)
            print(f"[Epoch {epoch:03d} | Iter {step:04d}] ",
                    f"Loss: {mean_loss:.4f} | ",
                    f"GradNorm: {avg_grad_norm:.6f} | ",
                    f"LR: {current_lr:.6f}",)
    train_loss = torch.mean(torch.tensor(loss_list))
    train_accs = acc_sum_tensor[0] / acc_sum_tensor[1]
    train_ious = iou_sum_tensor[0] / iou_sum_tensor[1]
    
    return train_loss, train_accs.cpu().numpy(), train_ious.cpu().numpy()

def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    loss_list = []
    acc_sum_tensor = torch.zeros((2, num_classes), dtype=torch.float64, device=device)
    iou_sum_tensor = torch.zeros((2, num_classes), dtype=torch.float64, device=device)
    with torch.no_grad():
        for i, input_dict in enumerate(tqdm(loader, desc='Validation', leave=False)):
            points = input_dict["points"].to(device)
            labels = input_dict["labels"].to(device)
            scores = model(points)
            loss = criterion(scores, labels)
            mean_loss = reduce_value(loss, average=True)            
            acc_tensor = reduce_value(get_accuracy_tensor(scores, labels, device=device), average=False)
            iou_tensor = reduce_value(get_iou_tensor(scores, labels, device=device), average=False)
            loss_list.append(mean_loss.cpu())
            acc_sum_tensor += acc_tensor
            iou_sum_tensor += iou_tensor
            
        val_loss = torch.mean(torch.tensor(loss_list))
        val_accs = acc_sum_tensor[0] / acc_sum_tensor[1]
        val_ious = iou_sum_tensor[0] / iou_sum_tensor[1]
        
    return val_loss, val_accs.cpu().numpy(), val_ious.cpu().numpy()

def reduce_value(value, average=True):
    world_size = int(os.environ.get('WORLD_SIZE', 1))    
    if world_size < 2:
        return value
    else:
        with torch.no_grad():
            dist.all_reduce(value)
            if average:
                value /= world_size
        return value

def train(args):
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)

    logs_dir = args.logs_dir / args.name
    if args.distributed:
        if args.rank == 0:
            logs_dir.mkdir(exist_ok=True, parents=True)
        dist.barrier()  # 等待rank 0创建完目录后，其他rank才能继续
    else:
        logs_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载数据集
    DATA = load_yaml(args.dataset_cfg)
    train_dataset = SemanticKITTIDataset(args.dataset, args.task, DATA, cfg, args.gpu, split="train")
    valid_dataset = SemanticKITTIDataset(args.dataset, args.task, DATA, cfg, args.gpu, split="valid")
    num_classes = train_dataset.num_classes

    if args.distributed:
        # 给每个rank对应的GPU进程分配训练的样本索引
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)
    else:
        # 非分布式训练，直接使用随机或顺序采样器
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(valid_dataset)

    # 将样本索引组成batch
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    val_batch_sampler = torch.utils.data.BatchSampler(val_sampler, args.batch_size, drop_last=False)

    trainLoader = torch.utils.data.DataLoader(train_dataset,
                                             batch_sampler=train_batch_sampler,    
                                             pin_memory=True,
                                             num_workers=args.num_workers,)

    validLoader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_sampler=val_batch_sampler,    
                                             pin_memory=True,
                                             num_workers=args.num_workers,)

    d_in = next(iter(train_dataset))["points"].size(-1)

    model = RandLANet(
        d_in,
        num_classes,
        num_neighbors=args.neighbors,
        decimation=args.decimation,
        device=args.gpu
    )

    if args.distributed:
        if args.rank == 0:
            print('Classes Weights:', train_dataset.weights)
            print("Initialized in main process")
        for name, param in model.state_dict().items():
            torch.distributed.broadcast(param, src=0)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        print('Classes Weights:', train_dataset.weights)

    # print('Computing weights...', end='\t')
    # samples_per_class = np.array(cfg.class_weights)
    # n_samples = torch.tensor(cfg.class_weights, dtype=torch.float, device=args.gpu)
    # ratio_samples = n_samples / n_samples.sum()
    # weights = 1 / (ratio_samples + 0.02)
    # print('Done.')
    
    criterion = nn.CrossEntropyLoss(weight=train_dataset.weights, ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.scheduler_gamma)

    first_epoch = 1
    if args.load:
        path = max(list((args.logs_dir / args.load).glob('*.pth')))
        print(f'Loading {path}...')
        checkpoint = torch.load(path)
        first_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if args.distributed:
        if args.rank == 0:
            writer = SummaryWriter(logs_dir)
        else:
            writer = None
    else:
        writer = SummaryWriter(logs_dir)

    for epoch in range(first_epoch, args.epochs+1):
        t0 = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        if writer != None:
            print(f'=== EPOCH {epoch:d}/{args.epochs:d} ===')
        
        train_loss, train_accs, train_ious = train_one_epoch(model=model,
                                                           optimizer=optimizer,
                                                           data_loader=trainLoader,
                                                           criterion=criterion,
                                                           writer=writer,
                                                           device=args.gpu,
                                                           num_classes=num_classes,
                                                           epoch=epoch)
        
        scheduler.step()
        val_loss, val_accs, val_ious = evaluate(model=model,
                                                loader=validLoader,
                                                criterion=criterion,
                                                device=args.gpu,
                                                num_classes=num_classes)
        if writer != None:
            loss_dict = {
                'Training loss':    train_loss,
                'Validation loss':  val_loss
            }
            acc_dicts = [
                {
                    'Training accuracy': acc,
                    'Validation accuracy': val_acc
                } for acc, val_acc in zip(train_accs, val_accs)
            ]
            iou_dicts = [
                {
                    'Training accuracy': iou,
                    'Validation accuracy': val_iou
                } for iou, val_iou in zip(train_ious, val_ious)
            ]

            t1 = time.time()
            d = t1 - t0
            # Display results
            for k, v in loss_dict.items():
                print(f'{k}: {v:.7f}', end='\t')
            print()
            
            print('Accuracy     ', *[f'{train_dataset.new_label_map[cls_idx]}' for cls_idx in range(num_classes)], '   OA', sep=' | ')
            print('Training:    ', *[f'{acc:.3f}' if not np.isnan(acc) else '  nan' for acc in train_accs], f'{np.nanmean(train_accs):.3f}', sep=' | ')
            print('Validation:  ', *[f'{acc:.3f}' if not np.isnan(acc) else '  nan' for acc in val_accs], f'{np.nanmean(val_accs):.3f}', sep=' | ')

            print('IoU          ', *[f'{train_dataset.new_label_map[cls_idx]}' for cls_idx in range(num_classes)], ' mIoU', sep=' | ')
            print('Training:    ', *[f'{iou:.3f}' if not np.isnan(iou) else '  nan' for iou in train_ious], f'{np.nanmean(train_ious):.3f}', sep=' | ')
            print('Validation:  ', *[f'{iou:.3f}' if not np.isnan(iou) else '  nan' for iou in val_ious], f'{np.nanmean(val_ious):.3f}', sep=' | ')

            print('Time elapsed:', '{:.0f} s'.format(d) if d < 60 else '{:.0f} min {:02.0f} s'.format(*divmod(d, 60)))

            # send results to tensorboard
            writer.add_scalars('Loss', loss_dict, epoch)

            for cls_idx in range(num_classes):
                cls_name = train_dataset.new_label_map[cls_idx]
                writer.add_scalars(f'Per-class accuracy/{cls_name}', acc_dicts[cls_idx], epoch)
                writer.add_scalars(f'Per-class IoU/{cls_name}', iou_dicts[cls_idx], epoch)
            writer.add_scalars('Per-class accuracy/Overall', acc_dicts[-1], epoch)
            writer.add_scalars('Per-class IoU/Mean IoU', iou_dicts[-1], epoch) 
            
            if epoch % args.save_freq == 0:
                torch.save(
                    dict(
                        epoch=epoch,
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict()
                    ),
                    args.logs_dir / args.name / f'checkpoint_{epoch:02d}.pth'
                )
            
    if writer is not None:
        writer.close()
        

if __name__ == '__main__':

    """Parse program arguments"""
    parser = argparse.ArgumentParser(
        prog='RandLA-Net',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    base = parser.add_argument_group('Base options')
    expr = parser.add_argument_group('Experiment parameters')
    param = parser.add_argument_group('Hyperparameters')
    dirs = parser.add_argument_group('Storage directories')
    misc = parser.add_argument_group('Miscellaneous')

    base.add_argument('--task', type=str, help='',
                        default='movable', choices=['semantic', 'movable', 'moving'])
    base.add_argument('--dataset', type=Path, help='location of the dataset',
                        default='datasets/s3dis/subsampled')
    base.add_argument('--dataset_cfg', type=Path, help='config of the dataset',
                        default='config/semantic-kitti-mos.yaml')
    
    expr.add_argument('--epochs', type=int, help='number of epochs',
                        default=50)
    expr.add_argument('--load', type=str, help='model to load',
                        default='')
    
    param.add_argument('--adam_lr', type=float, help='learning rate of the optimizer',
                        default=1e-2)
    param.add_argument('--batch_size', type=int, help='batch size',
                        default=1)
    param.add_argument('--decimation', type=int, help='ratio the point cloud is divided by at each layer',
                        default=4)
    param.add_argument('--dataset_sampling', type=str, help='how dataset is sampled',
                        default='naive', choices=['active_learning', 'naive'])
    param.add_argument('--neighbors', type=int, help='number of neighbors considered by k-NN',
                        default=16)
    param.add_argument('--scheduler_gamma', type=float, help='gamma of the learning rate scheduler',
                        default=0.95)

    dirs.add_argument('--logs_dir', type=Path, help='path to tensorboard logs',
                        default='runs')

    misc.add_argument('--local_rank', type=int, help='local rank for distributed training',
                        default=0)
    misc.add_argument('--gpu', type=int, help='which GPU to use (-1 for CPU)', 
                        default=0)
    misc.add_argument('--name', type=str, help='name of the experiment',
                        default=None)
    misc.add_argument('--num_workers', type=int, help='number of threads for loading data',
                        default=8)
    misc.add_argument('--save_freq', type=int, help='frequency of saving checkpoints',
                        default=4)

    args = parser.parse_args()

    if args.gpu >= 0:
        if torch.cuda.is_available():
            args.gpu = torch.device(f'cuda:{args.gpu:d}')
        else:
            warnings.warn('CUDA is not available on your machine. Running the algorithm on CPU.')
            args.gpu = torch.device('cpu')
    else:
        args.gpu = torch.device('cpu')

    if int(os.environ.get("DEBUG", "1")) == 0:
        args.distributed = True
        init_distributed_mode(args)
    else:
        args.distributed = False

    if args.name is None:
        if args.load:
            args.name = args.load
        else:
            args.name = datetime.now().strftime('%Y-%m-%d_%H:%M')

    t0 = time.time()
    train(args)
    t1 = time.time()

    d = t1 - t0
    print('Done. Time elapsed:', '{:.0f} s.'.format(d) if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))
