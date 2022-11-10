#!/usr/bin/env python

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# CUDA_VISIBLE_DEVICES=2 nohup python train.py -a vit_base_in21k --optimizer=adamw --lr=1e-6 --weight-decay=1e-1 --epochs=20 --warmup-epochs=0 --moco-m-cos --moco-t=.5 --pretrained --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --data-set cifar10 --moco-dim 1024 --ckpt /homes/yl4002/moco-v3-main/unsup_05_1024 --batch-size 64 > log_unsup_05_1024_wd_01.out 2>&1 &


# CUDA_VISIBLE_DEVICES=0 python train.py -a vit_base --image-size=32 --optimizer=adamw --lr=1e-6 --weight-decay=1e-1 --epochs=20 --warmup-epochs=0 --moco-m-cos --moco-t=.5 --pretrained --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --data-set cifar10 --moco-dim 768 --ckpt /homes/yl4002/moco-v3-main/unsup_05_768 --batch-size 256

# CUDA_VISIBLE_DEVICES=0,1,2,4 nohup python train.py -a vit_base --image-size=32 --optimizer=adamw --lr=1e-6 --weight-decay=1e-1 --epochs=20 --warmup-epochs=0 --moco-m-cos --moco-t=.5 --pretrained --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --data-set cifar10 --moco-dim 768 --ckpt /homes/yl4002/moco-v3-main/unsup_05_768 --batch-size 256 > log_unsup_05_768_wd_01.out 2>&1 &


import argparse
import builtins
import math
import os
import random
import shutil
import time
import datetime
import warnings
from functools import partial
from timm.utils import AverageMeter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

import moco.builder
import moco.loader
import moco.optimizer
from datasets import build_dataset
from logger import create_logger

from timm.models import deit_small_patch16_224
from timm.models import vit_small_patch16_224
from timm.models import vit_base_patch16_224
from timm.models import vit_base_patch16_224_in21k
from timm.models import swin_base_patch4_window7_224
from timm.models import swin_base_patch4_window7_224_in22k

from timm.models import resnetv2_50

available_models = dict(deit_small=deit_small_patch16_224, vit_small=vit_small_patch16_224,
                        vit_base=vit_base_patch16_224, vit_base_in21k=vit_base_patch16_224_in21k,
                        swin_base=swin_base_patch4_window7_224, swin_base_in21k=swin_base_patch4_window7_224_in22k,
                        resnet50=resnetv2_50)

parser = argparse.ArgumentParser(description='OOD Detection Pre-Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_base', help='model architecture: (default: vit_base)')
parser.add_argument('-j', '--workers', default=15, type=int, metavar='N',
                    help='number of data loading workers (default: 15)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--data-path', default='./data', type=str, help='dataset path')
parser.add_argument('--data-set', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'], type=str,
                    help='Dataset name')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2048, type=int, metavar='N',
                    help='mini-batch size (default: 2048), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--image-size', default=224, type=int, metavar='N', help='image size (default: 224)')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float, metavar='LR',
                    help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay (default: 1e-6)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int, help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int, help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true', help='gradually increase moco momentum to 1 with a '
                                                              'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=1.0, type=float, help='softmax temperature (default: 1.0)')

# other upgrades
parser.add_argument('--optimizer', default='lars', type=str, choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N', help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float, help='minimum scale for random cropping (default: 0.08)')
parser.add_argument('--log_dir', default='output', type=str, help='path to log')
parser.add_argument('--ckpt', default='', type=str, help='absolute path folder to pretrained checkpoints',
                    required=True)
parser.add_argument('--pretrained', action='store_true', help='whether use pretrained model')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node', ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)
        torch.distributed.barrier()
        logger = create_logger(output_dir=args.log_dir, dist_rank=dist.get_rank(), name=f"ood")
    else:
        logger = create_logger(output_dir=args.log_dir, dist_rank=0, name=f"ood")
    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    logger.info("=> pretrained '{}'".format(args.pretrained))
    model = moco.builder.MoCo(available_models[args.arch], args.moco_dim, args.moco_mlp_dim, args.image_size, args.moco_t,
                              pretrained=args.pretrained)

    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # comment output the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # print(model)  # print model after SyncBatchNorm

    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(model.parameters(), args.lr, weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler()
    summary_writer = SummaryWriter() if args.rank == 0 else None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # print(checkpoint['optimizer'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_dataset, nb_classes = build_dataset(True, args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None), num_workers=args.workers,
                                               pin_memory=True, sampler=train_sampler, drop_last=True)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args, logger)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.gpu == 0):  # only the first GPU saves checkpoint
            torch.save({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), },
                       os.path.join(args.ckpt, 'checkpoint_%04d.pth' % epoch))

    if args.gpu == 0:
        summary_writer.close()


def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args, logger):
    batch_time = AverageMeter()
    learning_rates = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    end = time.time()

    num_steps = len(train_loader)
    moco_m = args.moco_m
    for i, (images, _) in enumerate(train_loader):

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / num_steps, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / num_steps, args)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss = model(images[0], images[1], moco_m)

        losses.update(loss.item(), images[0].size(0))
        if args.gpu == 0:
            summary_writer.add_scalar("loss", loss.item(), epoch * num_steps + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - i)
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{i}/{num_steps}]\t' + f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.10f}\t' + f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t' + f'loss {losses.val:.6f} ({losses.avg:.6f})\t' + f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (
                1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


if __name__ == '__main__':
    main()
