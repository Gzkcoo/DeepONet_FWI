import numpy as np
import torch
import time
import torch.nn as nn
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data.dataloader import default_collate
from timeit import default_timer
import utils
import cv2
import json
import sys
import os
import transforms as T
import pytorch_ssim
import network
import datetime
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torchvision.transforms import Compose
from dataset import LDataset, FLDataset
from vis import *

def evaluate(model, criterion, dataloader, device, k, ctx,
                vis_path, vis_batch, vis_sample, missing, std, with_source):
    model.eval()

    label_list, label_pred_list= [], [] # store denormalized predcition & gt in numpy
    label_tensor, label_pred_tensor = [], [] # store normalized prediction & gt in tensor

    with torch.no_grad():
        batch_idx = 0
        for data, source, label in dataloader:
            data = data.to(device, non_blocking=True)
            source = source.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)



            label_np = T.tonumpy_denormalize(label, ctx['label_min'], ctx['label_max'], exp=False)
            label_list.append(label_np)
            label_tensor.append(label)

            if with_source == 1:
                pred = model(data, source)
            else:
                pred = model(data)

            label_pred_np = T.tonumpy_denormalize(pred, ctx['label_min'], ctx['label_max'], exp=False)
            label_pred_list.append(label_pred_np)
            label_pred_tensor.append(pred)

            # Visualization
            if vis_path and batch_idx < vis_batch:
                for i in range(vis_sample):
                    plot_velocity(label_pred_np[i, 0], label_np[i, 0],
                                  f'{vis_path}/V_{batch_idx}_{i}.png')  # , vmin=ctx['label_min'], vmax=ctx['label_max'])
            batch_idx += 1
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    label, label_pred = np.concatenate(label_list), np.concatenate(label_pred_list)
    label_t, pred_t = torch.cat(label_tensor), torch.cat(label_pred_tensor)

    print(f'MAE: {l1(label_t, pred_t)}')
    print(f'MSE: {l2(label_t, pred_t)}')
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    print(f'SSIM: {ssim_loss(label_t / 2 + 0.5, pred_t / 2 + 0.5)}') # (-1, 1) to (0, 1)

def test(args):

    utils.mkdir(args.output_path)
    torch.backends.cudnn.benchmark = True

    print(args)

    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[args.dataset]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    ctx['file_size'] = 500
    # Normalize data and label to [-1, 1]
    transform_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(T.log_transform(ctx['data_min'], k=args.k), T.log_transform(ctx['data_max'], k=1))
    ])
    transform_source = Compose([
        T.MinMaxNormalize(0, 700)
    ])
    transform_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])


    print('Loading validation data')

    dataset_all = LDataset(
        args.val_anno,
        preload=True,
        sample_ratio=1,
        file_size=ctx['file_size'],
        transform_data=transform_data,
        transform_source=transform_source,
        transform_label=transform_label,
        du=args.data_url
    )

    dataset_train, dataset_valid = torch.utils.data.random_split(dataset=dataset_all,
                                                                 lengths=[int(len(dataset_all) / 9 * 8),                                                               int(len(dataset_all) / 9)],
                                                                 generator=torch.Generator().manual_seed(0))                                                                 # Normalize data and label to [-1, 1]
    # dataset_valid = LDataset(
    #     args.val_anno,
    #     preload=True,
    #     sample_ratio=1,
    #     file_size=ctx['file_size'],
    #     transform_data=transform_data,
    #     transform_source=transform_source,
    #     transform_label=transform_label,
    #     du=args.data_url
    # )

    valid_sampler = SequentialSampler(dataset_valid)


    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate)

    print('Creating model')
    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()
    model = network.model_dict[args.model](sample_spatial=1.0, sample_temporal=1, layer_sizes_t=args.layer_sizes).to(args.device)

    # Define loss function
    l1loss = nn.L1Loss()  # MAE
    l2loss = nn.MSELoss()  # MSE

    # l3loss = network.BCEFocalLoss()  # 交叉熵

    def criterion(pred, gt):
        loss_g1v = l1loss(pred, gt)
        loss_g2v = l2loss(pred, gt)

        loss = args.lambda_g1v * loss_g1v + args.lambda_g2v * loss_g2v
        return loss, loss_g1v, loss_g2v

    if args.resume:
        print(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(network.replace_legacy(checkpoint['model']))
        print('Loaded model checkpoint at Epoch {} / Step {}.'.format(checkpoint['epoch'], checkpoint['step']))

    if args.vis:
        # Create folder to store visualization results
        vis_folder = f'visualization_{args.vis_suffix}' if args.vis_suffix else 'visualization'
        vis_path = os.path.join(args.train_out, vis_folder)
        utils.mkdir(vis_path)
    else:
        vis_path = None


    print('Start testing')
    evaluate(model, criterion, dataloader_valid, args.device, args.k, ctx,
             vis_path, args.vis_batch, args.vis_sample, args.missing, args.std, args.with_source)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Testing')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='flatvel-b-L', type=str, help='dataset name')
    parser.add_argument('-fs', '--file-size', default=None, type=int, help='number of samples in each npy file')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='split_files', help='annotation files location')
    parser.add_argument('-v', '--val_anno', default='flatvel_a_val_L.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='Invnet_models', help='path to parent folder to save checkpoints')
    parser.add_argument('-n', '--save-name', default='fcn_l1loss_ffb', help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')

    # Model related
    parser.add_argument('--with_source', default=1, type=int, help='data with source or not')
    parser.add_argument('-m', '--model', default='InversionDeepOnet', type=str, help='inverse model name')
    parser.add_argument('-no', '--norm', default='bn', help='normalization layer type, support bn, in, ln (default: bn)')
    parser.add_argument('-um', '--up-mode', default=None, help='upsampling layer mode such as "nearest", "bicubic", etc.')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')
    parser.add_argument('--layer_sizes', type=int, nargs='+', default=[5, 256, 256, 256, 512], help='trunk_net layer sizes')

    # Test related
    parser.add_argument('-b', '--batch_size', default=128, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('-r', '--resume', default='model-out/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--vis', help='visualization option', action="store_false")
    parser.add_argument('-vsu','--vis-suffix', default=None, type=str, help='visualization suffix')
    parser.add_argument('-vb','--vis-batch', help='number of batch to be visualized', default=2, type=int)
    parser.add_argument('-vsa', '--vis-sample', help='number of samples in a batch to be visualized', default=3, type=int)
    parser.add_argument('--missing', default=0, type=int, help='number of missing traces')
    parser.add_argument('--std', default=0, type=float, help='standard deviation of gaussian noise')

    # nuaa
    parser.add_argument('--data_url', type=str, default='')  # data_url
    parser.add_argument('--train_model_out', type=str, default='')  # 模型输出路径
    parser.add_argument('--train_out', type=str, default='')  # 文件输出路径
    parser.add_argument('--train_visualized_log', type=str, default='')  # 可视化路径日志
    parser.add_argument('--gpu_num_per_node', type=int, default=1)  # GPU节点个数
    parser.add_argument('--model_load_dir', type=str, default='')  # 预训练模型路径


    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    args.resume = os.path.join(args.model_load_dir, args.resume)



    return args


if __name__ == '__main__':
    args = parse_args()
    start_time = default_timer()
    test(args)
    end_time = default_timer()
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(end_time-start_time)))))

