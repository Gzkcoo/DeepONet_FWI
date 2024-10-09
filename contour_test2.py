

# 两个输出口的deeponet

import os
import sys
import time
import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import default_collate
import torchvision
from torchvision.transforms import Compose
import numpy as np

import utils
import network
from vis import *
from dataset import LDataset
import transforms as T
import pytorch_ssim


def evaluate(model, criterions, dataloader, device, k, ctx,
             vis_path, vis_batch, vis_sample, missing, std):
    model.eval()

    label_list, label_pred_list = [], []  # store denormalized predcition & gt in numpy
    label_tensor, label_pred_tensor = [], []  # store normalized prediction & gt in tensor
    con_pred_tensor =  []  # 存储轮廓预测输出 in tensor
    con_tensor = []  # 存储轮廓真实值 in tensor

    with torch.no_grad():
        batch_idx = 0
        for data, source, label in dataloader:


            # label二值化处理
            # ------------------------------
            # conlabels = np.zeros((label.shape[0], 1, 70, 70))
            # for i in range(label.shape[0]):
            #     for j in range(label.shape[1]):
            #         conlabels[i, j, ...] = utils.extract_contours(label[i, j, ...])

            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            label_np = T.tonumpy_denormalize(label, ctx['label_min'], ctx['label_max'], exp=False)
            label_list.append(label_np)
            label_tensor.append(label)


            pred, conpred = model(data)

            con_pred_tensor.append(conpred)  # 轮廓预测输出

            label_pred_np = T.tonumpy_denormalize(pred, ctx['label_min'], ctx['label_max'], exp=False)
            label_pred_list.append(label_pred_np)
            label_pred_tensor.append(pred)


            # # Visualization
            # if vis_path and batch_idx < vis_batch:
            #     for i in range(vis_sample):
            #         plot_velocity(label_pred_np[i, 0], label_np[i, 0],
            #                       f'{vis_path}/V_{batch_idx}_{i}.png')  # , vmin=ctx['label_min'], vmax=ctx['label_max'])
            #         plot_single_velocity(conpred_np1[i, 0],
            #                       f'{vis_path}/V_{batch_idx}_{i}_con1.png')
            #         plot_single_velocity(conpred_np2[i, 0],
            #                       f'{vis_path}/V_{batch_idx}_{i}_con2.png')
            #         plot_single_velocity(conlabels[i, 0],
            #                              f'{vis_path}/V_{batch_idx}_{i}_con3.png')
            #
            #         if missing or std:
            #             for ch in [2]:  # range(data.shape[1]):
            #                 plot_seismic(data_np[i, ch], data_noise_np[i, ch], f'{vis_path}/S_{batch_idx}_{i}_{ch}.png',
            #                              vmin=ctx['data_min'] * 0.01, vmax=ctx['data_max'] * 0.01)
            # batch_idx += 1

            # conlabels = torch.tensor(conlabels).reshape((-1, 1)).to(device)
            # con_tensor.append(conlabels)  # 轮廓真实值

    label, label_pred = np.concatenate(label_list), np.concatenate(label_pred_list)
    label_t, pred_t = torch.cat(label_tensor), torch.cat(label_pred_tensor)
    # con_t, con_pred_t = torch.cat(con_tensor), torch.cat(con_pred_tensor)





    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    l3 = nn.CrossEntropyLoss()  # 交叉熵
    print(f'MAE: {l1(label_t, pred_t)}')
    print(f'MSE: {l2(label_t, pred_t)}')
    #print(f'CrossEntropy: {l3(con_pred_t, con_t.squeeze().long())}')

    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    print(f'SSIM: {ssim_loss(label_t / 2 + 0.5, pred_t / 2 + 0.5)}')  # (-1, 1) to (0, 1)


def main(args):
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    utils.mkdir(args.output_path)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[args.dataset]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    if args.file_size is not None:
        ctx['file_size'] = args.file_size

    print("Loading data")
    print("Loading validation data")
    log_data_min = T.log_transform(ctx['data_min'], k=args.k)
    log_data_max = T.log_transform(ctx['data_max'], k=args.k)
    transform_valid_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(log_data_min, log_data_max),
    ])

    transform_valid_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])
    if args.val_anno[-3:] == 'txt':
        dataset_valid = LDataset(
            args.val_anno,
            sample_ratio=args.sample_temporal,
            file_size=ctx['file_size'],
            transform_data=transform_valid_data,
            transform_label=transform_valid_label,
            du=args.data_url
        )
    else:
        dataset_valid = torch.load(args.val_anno)

    print("Creating data loaders")
    valid_sampler = SequentialSampler(dataset_valid)
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate)

    print("Creating model")
    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()

    model = network.model_dict[args.model](upsample_mode=args.up_mode,
                                           sample_spatial=args.sample_spatial, sample_temporal=args.sample_temporal,
                                           norm=args.norm, deepth=70, length=70).to(device)

    criterions = {
        'MAE': lambda x, y: np.mean(np.abs(x - y)),
        'MSE': lambda x, y: np.mean((x - y) ** 2)
    }

    if args.resume:
        print(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(network.replace_legacy(checkpoint['model']))
        print('Loaded model checkpoint at Epoch {} / Step {}.'.format(checkpoint['epoch'], checkpoint['step']))

    if args.vis:
        # Create folder to store visualization results
        vis_folder = f'visualization_{args.vis_suffix}' if args.vis_suffix else 'visualization'
        vis_path = os.path.join(args.output_path, vis_folder)
        utils.mkdir(vis_path)
    else:
        vis_path = None

    print("Start testing")
    start_time = time.time()
    evaluate(model, criterions, dataloader_valid, device, args.k, ctx,
             vis_path, args.vis_batch, args.vis_sample, args.missing, args.std)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Testing')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='flatvel-a', type=str, help='dataset name')
    parser.add_argument('-fs', '--file-size', default=None, type=int, help='number of samples in each npy file')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='split_files', help='annotation files location')
    parser.add_argument('-v', '--val_anno', default='flatvel_a_val_origin.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='Invnet_models',
                        help='path to parent folder to save checkpoints')
    parser.add_argument('-n', '--save-name', default='fcn_l1loss_ffb', help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')

    # Model related
    parser.add_argument('-m', '--model', default='DDNet70Model', type=str, help='inverse model name')
    parser.add_argument('-no', '--norm', default='bn',
                        help='normalization layer type, support bn, in, ln (default: bn)')
    parser.add_argument('-um', '--up-mode', default=None,
                        help='upsampling layer mode such as "nearest", "bicubic", etc.')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')

    # Test related
    parser.add_argument('-b', '--batch_size', default=128, type=int)
    parser.add_argument('-j', '--workers', default=2, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('-r', '--resume', default='model-out/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--vis', help='visualization option', action="store_false")
    parser.add_argument('-vsu', '--vis-suffix', default=None, type=str, help='visualization suffix')
    parser.add_argument('-vb', '--vis-batch', help='number of batch to be visualized', default=2, type=int)
    parser.add_argument('-vsa', '--vis-sample', help='number of samples in a batch to be visualized', default=3,
                        type=int)
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
    main(args)
