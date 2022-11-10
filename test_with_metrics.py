#!/usr/bin/env python
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
from datasets import build_dataset
import numpy as np
import moco.builder
import moco.loader
import moco.optimizer
from metrics import calc_metrics
from logger import create_logger
import faiss
import matplotlib.pyplot as plt

from timm.models import deit_small_patch16_224
from timm.models import vit_small_patch16_224
from timm.models import vit_base_patch16_224
from timm.models import vit_base_patch16_224_in21k
from timm.models import swin_base_patch4_window7_224
from timm.models import swin_base_patch4_window7_224_in22k
from timm.models import resnetv2_50

vit_models = dict(deit_small=deit_small_patch16_224, vit_small=vit_small_patch16_224, vit_base=vit_base_patch16_224,
                  vit_base_in21k=vit_base_patch16_224_in21k, swin_base=swin_base_patch4_window7_224,
                  swin_base_in21k=swin_base_patch4_window7_224_in22k, resnet50=resnetv2_50)

parser = argparse.ArgumentParser(description='PyTorch OOD detection metric')
parser.add_argument('--data-path', default='./data', type=str, help='dataset path')
parser.add_argument('--data-set', default='cifar10', choices=['cifar10', 'cifar100'], type=str, help='train dataset')
parser.add_argument('--ood-data-set', default=None, choices=['cifar10', 'cifar100', 'svhn', 'lsun', 'place365'],
                    type=str, help='ood dataset, if empty, the list of choices will be processed')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_base', help='model architecture: (default: vit_base)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--image-size', default=224, type=int, metavar='N', help='image size (default: 224)')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
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
parser.add_argument('--moco-t', default=0.07, type=float, help='softmax temperature (default: 0.07)')

# additional configs:
parser.add_argument('--ckpt', default='', type=str, help='absolute path folder to pretrained checkpoints',
                    required=True)
parser.add_argument('--log_dir', default='output', type=str, help='path to log')
parser.add_argument('--mode', default='unsup', choices=['sup', 'unsup'], type=str, help='unsupervised or supervised')
parser.add_argument('--feature-type', default='ensemble', choices=['ensemble', 'encoder', 'predictor'], type=str,
                    help='Feature type')
parser.add_argument('--clusters', default=1, type=int, help='unsupervised or supervised')
parser.add_argument('--plot_debug', action='store_true', help='plot figures and save the features')


# CUDA_VISIBLE_DEVICES=2 nohup python test_with_metrics.py -a vit_base --gpu 0 --data-set cifar10 --feature-type=ensemble --plot_debug --ckpt /homes/yl4002/moco-v3-main/unsup_05_768 --moco-dim=768 --mode unsup --clusters 1 --batch-size 256 > log_metric_unsup_05_768_1.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python test_with_metrics.py -a swin_base --gpu 0 --data-set cifar10 --feature-type=predictor --ckpt /homes/yl4002/moco-v3-main/unsup_05_768_cifar10_swin --moco-dim=768 --mode unsup --clusters 1 --batch-size 256 > log_metric_unsup_05_768_cifar10_swin_1.out 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1 nohup python test_with_metrics.py -a vit_base_in21k --gpu 0 --data-set cifar10 --feature-type=ensemble --ckpt /homes/yl4002/moco-v3-main/unsup_05_1024 --moco-dim=1024 --mode unsup --clusters 1 --batch-size 256 > log_metric_unsup_05_1024_cifar10_swin_1.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0,1 nohup python test_with_metrics.py -a vit_base --gpu 0 --data-set cifar10 --image-size=32 --feature-type=ensemble --ckpt /homes/yl4002/moco-v3-main/unsup_05_768 --moco-dim=768 --mode unsup --clusters 1 --batch-size 256 > log_metric_unsup_05_768_cifar10_1.out 2>&1 &

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

    if os.path.isfile(os.path.join(args.ckpt, 'metrics.txt')):  # earlier Auroc file exist then remove it
        os.remove(os.path.join(args.ckpt, 'metrics.txt'))

    ngpus_per_node = torch.cuda.device_count()
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

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for testing".format(args.gpu))
    os.makedirs(args.log_dir, exist_ok=True)

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

    files = os.listdir(args.ckpt)
    files = [os.path.join(args.ckpt, f) for f in files]  # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    logger.info(f'{len(files)} files in {args.ckpt}')
    id_dataset_name = args.data_set
    for file in files:
        if not file.endswith('.pth.tar') and not file.endswith('.pth'):
            continue
        args.pretrained = file
        run_ood_eval(args, id_dataset_name, logger)


def run_ood_eval(args, id_dataset_name, logger):
    ngpus_per_node = torch.cuda.device_count()
    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(vit_models[args.arch], args.moco_dim, args.moco_mlp_dim, args.image_size, args.moco_t)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            logger.info("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cuda:0")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            msg = model.load_state_dict(state_dict, strict=False)
            logger.info(msg)
            logger.info("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.pretrained))
    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif args.distributed:
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
        model = model.cuda(args.gpu)

    cudnn.benchmark = True

    # Data loading code
    args.data_set = id_dataset_name
    train_dataset, train_classes = build_dataset(True, args, False)
    train_dataset_len = len(train_dataset)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None), num_workers=args.workers,
                                               pin_memory=True, sampler=train_sampler)
    logger.info(
        f'run the model on ID training dataset to get features. Dataset:{id_dataset_name},length: {train_dataset_len}')
    train_features, train_targets = run_model(train_loader, model, args, logger)

    if args.mode == 'unsup':
        logger.info(f'kmeans classify the training data into {args.clusters} classes.')
        kmeans = faiss.Kmeans(train_features.shape[1], args.clusters, niter=100, verbose=False, gpu=False)
        kmeans.train(train_features.numpy())
        _, train_targets = kmeans.index.search(train_features.numpy(), 1)
        train_targets = torch.from_numpy(train_targets)
        train_targets = train_targets.squeeze(1)

    in_classes = torch.unique(train_targets)
    class_idx = [torch.nonzero(torch.eq(cls, train_targets)).squeeze(dim=1) for cls in in_classes]
    classes_feats = [train_features[idx] for idx in class_idx]
    logger.info(f'compute the mean and std of the features in {args.data_set}.')
    classes_mean = torch.stack([torch.mean(cls_feats, dim=0) for cls_feats in classes_feats], dim=0)
    sup_inv_cov = [np.linalg.inv(np.cov(cls_feats, rowvar=False)) for cls_feats in classes_feats]

    # ood_dataset_list = ['cifar10', 'cifar100', 'svhn', 'lsun', 'place365']
    ood_dataset_list = ['cifar10', 'cifar100', 'svhn']
    if args.ood_data_set is not None:
        ood_dataset_list = [args.ood_data_set]
    else:
        ood_dataset_list.remove(args.data_set)

    test_dataset, test_classes = build_dataset(False, args, False)
    id_dataset_len = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers, pin_memory=True)
    logger.info(
        f'run the model on ID testing dataset to get features. Dataset:{id_dataset_name},length: {id_dataset_len}')
    in_features, in_targets = run_model(test_loader, model, args, logger)
    if args.plot_debug:
        torch.save(in_features, os.path.join(args.ckpt, f'{id_dataset_name}.pt'))

    logger.info(f'calc the mahalanobis distance of ID testing dataset. Dataset:{id_dataset_name},')
    in_dists = mahalanobis(in_features, classes_mean, sup_inv_cov)

    in_scores_mahalanobis = torch.max(in_dists, dim=1).values
    for ood_dataset_name in ood_dataset_list:
        logger.info(f'process ood {ood_dataset_name}')
        args.data_set = ood_dataset_name
        ood_dataset, _ = build_dataset(False, args, False)
        ood_dataset_len = len(ood_dataset)
        ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True, )
        logger.info(
            f'run the model on OOD dataset to get features. Dataset:{ood_dataset_name},length: {ood_dataset_len}')
        ood_features, ood_targets = run_model(ood_loader, model, args, logger)
        if args.plot_debug:
            torch.save(ood_features, os.path.join(args.ckpt, f'{ood_dataset_name}.pt'))
        logger.info(f'calc the mahalanobis distance of ID testing dataset. Dataset:{ood_dataset_name},')

        batch = 1
        if ood_dataset_len > 10000:
            batch = (ood_dataset_len // 10000) + 1

        ood_dists_list = []
        for i in range(batch):
            start = 10000 * i
            end = 10000 * (i + 1)
            if batch == i + 1:
                end = ood_dataset_len
            batch_features = ood_features[start:end]
            ood_dists_list.append(mahalanobis(batch_features, classes_mean, sup_inv_cov))
        ood_dists = torch.cat(ood_dists_list)

        # calculating output distribution scores which are closest to ID
        ood_scores_mahalanobis = torch.max(ood_dists, dim=1).values
        labels = [1] * len(in_scores_mahalanobis) + [0] * len(ood_scores_mahalanobis)
        scores = np.concatenate((in_scores_mahalanobis, ood_scores_mahalanobis))
        metrics = calc_metrics(scores, labels)
        metrics['model'] = args.pretrained[args.pretrained.rindex('/') + 1:]
        metrics['id_name'] = id_dataset_name
        metrics['ood_name'] = ood_dataset_name
        logger.info('Mahalanobis OOD Metrics:{0}'.format(metrics))
        if args.plot_debug:
            plot_distribution(in_scores_mahalanobis, ood_scores_mahalanobis, id_dataset_name, ood_dataset_name, args)
        with open(os.path.join(args.ckpt, id_dataset_name + '_' + ood_dataset_name + '_metrics.txt'), "a") as f:
            f.write('{0}\n'.format(metrics))


def plot_distribution(in_scores, ood_scores, id_name, ood_name, args):
    bins_in = in_scores.shape[0] // 100
    bins_ood = ood_scores.shape[0] // 100
    plt.hist(in_scores.numpy(), bins=bins_in, density=False, alpha=0.6, label='ID:' + id_name)
    plt.hist(ood_scores.numpy(), bins=bins_ood, density=False, alpha=0.6, label='OOD:' + ood_name)
    plt.xlabel('distance')
    plt.ylabel('samples')
    plt.legend()
    plt.savefig(os.path.join(args.ckpt, f'{id_name}_{ood_name}_dist.png'))
    plt.clf()
    torch.save(in_scores, os.path.join(args.ckpt, f'{id_name}_{ood_name}_dist_{id_name}.pt'))
    torch.save(ood_scores, os.path.join(args.ckpt, f'{id_name}_{ood_name}_dist_{ood_name}.pt'))


@torch.no_grad()
def run_model(data_loader, model, args, logger):
    model.eval()

    num_steps = len(data_loader)
    batch_time = AverageMeter()

    start = time.time()
    end = time.time()

    train_features = []
    train_targets = []

    for i, (images, target) in enumerate(data_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()

        if args.feature_type == 'ensemble':
            output_encoder = model.base_encoder(images)
            output_pred = model.predictor(model.projector(output_encoder))
            output = (output_encoder + output_pred) / 2
        elif args.feature_type == 'predictor':
            output = model.predictor(model.projector(model.base_encoder(images)))
        else:
            output = model.base_encoder(images)
        output = nn.functional.normalize(output, dim=1)
        train_features.append(output.cpu())
        train_targets.append(target.cpu())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - i)
            logger.info(f'Eval: [{i}/{num_steps}]\t'
                        f'eta {datetime.timedelta(seconds=int(etas))} \t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"Eval takes {datetime.timedelta(seconds=int(epoch_time))}")
    return torch.cat(train_features), torch.cat(train_targets)


def mahalanobis(x, support_mean, inv_covmat):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    # create function to calculate Mahalanobis distance
    n = x.size(0)
    d = x.size(1)

    maha_dists = []
    for class_inv_cov, support_class in zip(inv_covmat, support_mean):
        support_class = support_class.to(device)
        x_mu = x - support_class.unsqueeze(0).expand(n, d)
        class_inv_cov = torch.from_numpy(class_inv_cov).float()
        class_inv_cov = class_inv_cov.to(device)
        left = torch.matmul(x_mu, class_inv_cov)
        mahal = torch.matmul(left, x_mu.transpose(0, 1))
        mahal = torch.diagonal(mahal, 0)
        maha_dists.append(mahal)
    res = -torch.cat(maha_dists).view(n, -1)
    return res.cpu()


if __name__ == '__main__':
    main()
