import numpy as np
import torch
import time
import torch.nn as nn
from torchvision import models
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
from scheduler import WarmupMultiStepLR
import network
import datetime
from torch.utils.data import RandomSampler, DataLoader
from torchvision.transforms import Compose
from dataset import LDataset
import matplotlib.pyplot as plt


step = 0
loss_history_MAE = []
loss_history_MSE = []

loss_history_val_MAE  = []
loss_history_val_MSE  = []

def train_one_epoch(model, criterion, optimizer, lr_scheduler,
                    dataloader, device, epoch, with_source):
    global step
    global loss_history_MAE
    global loss_history_MSE
    model.train()

    train_l1 = 0
    train_mse = 0


    t1 = default_timer()
    for data, source, label in dataloader:
        optimizer.zero_grad()
        data, source, label = data.to(device), source.to(device), label.to(device)
        if with_source == 1:
            output = model(data, source)
        else:
            output = model(data)
        loss, loss_g1v, loss_g2v = criterion(output, label)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_g1v_val = loss_g1v.item()
        loss_g2v_val = loss_g2v.item()


        step += 1
        lr_scheduler.step()
        train_l1 += loss_g1v_val
        train_mse += loss_g2v_val


    train_l1 /= len(dataloader)
    train_mse /= len(dataloader)

    t2 = default_timer()
    loss_history_MAE.append(train_l1)
    loss_history_MSE.append(train_mse)
    print('Train epoch {:d} , L1Loss = {:.6f}, MSELoss = {:.6f}, using {:.6f}s'.format(
        epoch, train_l1, train_mse, t2 - t1))


def evaluate(model, criterion, dataloader, device, with_source):
    model.eval()
    global loss_history_val_MAE
    global loss_history_val_MSE

    test_loss = 0
    test_l1 = 0
    test_mse = 0


    with torch.no_grad():
        label_tensor, label_pred_tensor = [], []  # store normalized prediction & gt in tensorS
        t1 = default_timer()
        for data, source, label in dataloader:
            data = data.to(device, non_blocking=True)
            source = source.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            if with_source == 1:
                output = model(data, source)
            else:
                output = model(data)
            # label_tensor.append(label)
            # label_pred_tensor.append(output)

            loss, loss_g1v, loss_g2v = criterion(output, label)

            test_l1 += loss_g1v.item()
            test_mse += loss_g2v.item()

    test_loss /= len(dataloader)
    test_l1 /= len(dataloader)
    test_mse /= len(dataloader)

    t2 = default_timer()

    loss_history_val_MAE.append(test_l1)
    loss_history_val_MSE.append(test_mse)

    print('Test L1Loss = {:.6f}, MSELoss = {:.6f}, using {:.6f}s'.format(
        test_l1, test_mse,  t2 - t1))
    return test_l1

def train(args):

    global loss_history_val_MSE
    global loss_history_val_MAE
    global loss_history_MSE
    global loss_history_MAE

    print(args)

    utils.mkdir(args.output_path)
    torch.backends.cudnn.benchmark = True

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

    print('Loading training data')\


    dataset_all = LDataset(
        args.train_anno,
        preload=True,
        sample_ratio=1,
        file_size=ctx['file_size'],
        transform_data=transform_data,
        transform_source=transform_source,
        transform_label=transform_label,
        du=args.data_url
    )

    dataset_train, dataset_valid = torch.utils.data.random_split(dataset=dataset_all,
                                                                 lengths=[int(len(dataset_all) / 9 * 8),
                                                                          int(len(dataset_all) / 9)],
                                                                 generator=torch.Generator().manual_seed(0))

    # Normalize data and label to [-1, 1]
    # dataset_train = MyDataset(
    #     args.train_anno,
    #     preload=True,
    #     sample_ratio=1,
    #     file_size=ctx['file_size'],
    #     transform_data=transform_data,
    #     transform_source=transform_source,
    #     transform_label=transform_label,
    #     du=args.data_url
    # )
    #
    # print('Loading validation data')
    # # Normalize data and label to [-1, 1]
    # dataset_valid = MyDataset(
    #     args.val_anno,
    #     preload=True,
    #     sample_ratio=1,
    #     file_size=ctx['file_size'],
    #     transform_data=transform_data,
    #     transform_source=transform_source,
    #     transform_label=transform_label,
    #     du = args.data_url
    # )

    train_sampler = RandomSampler(dataset_train)
    valid_sampler = RandomSampler(dataset_valid)


    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, drop_last=True, collate_fn=default_collate)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # Convert scheduler to be per iteration instead of per epoch
    warmup_iters = args.lr_warmup_epochs * len(dataloader_train)
    lr_milestones = [len(dataloader_train) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    print('Start training')
    best_loss = 10
    chp = 1

    for epoch in range(0, args.epoch):
        train_one_epoch(model, criterion, optimizer, lr_scheduler, dataloader_train,
                        args.device, epoch, args.with_source)

        loss = evaluate(model, criterion, dataloader_valid, args.device, args.with_source)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'step': step}

        # Save checkpoint per epoch
        if loss < best_loss:
            utils.save_on_master(
                checkpoint,
                # os.path.join(args.output_path, 'checkpoint.pth'))
                os.path.join(args.train_model_out, 'checkpoint.pth'))  # nuaa
            print('saving checkpoint at epoch: ', epoch)
            chp = epoch
            best_loss = loss
        # Save checkpoint every epoch block
        print('current best loss: ', best_loss)
        print('current best epoch: ', chp)

    # loss可视化
    # 绘制loss曲线图
    plt.figure(1)
    ax11 = plt.subplot(1, 1, 1)
    ax11.semilogy(np.arange(1, args.epoch+1), loss_history_MAE, label='train')
    ax11.semilogy(np.arange(1, args.epoch+1), loss_history_val_MAE, label='test')
    ax11.set_xlabel('Epochs')
    ax11.set_ylabel('Loss')
    ax11.set_title('MAELoss')
    ax11.legend()

    plt.savefig(os.path.join(args.train_out, "MAE.png"))

    plt.figure(2)
    ax21 = plt.subplot(1, 1, 1)
    ax21.semilogy(np.arange(1, args.epoch + 1), loss_history_MSE, label='train')
    ax21.semilogy(np.arange(1, args.epoch + 1), loss_history_val_MSE, label='test')
    ax21.set_xlabel('Epochs')
    ax21.set_ylabel('Loss')
    ax21.set_title('MSELoss')
    ax21.legend()

    plt.savefig(os.path.join(args.train_out, "MSE.png"))







def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Training')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='curvefault-a-L', type=str, help='dataset name')
    parser.add_argument('-fs', '--file-size', default=None, type=int, help='number of samples in each npy file')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='split_files', help='annotation files location')
    parser.add_argument('-t', '--train_anno', default='curvefault_a_train_L_all.txt', help='name of train anno')
    parser.add_argument('-v', '--val_anno', default='curvefault_a_val_L.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='Invnet_models',
                        help='path to parent folder to save checkpoints')
    parser.add_argument('-l', '--log-path', default='Invnet_models', help='path to parent folder to save logs')
    parser.add_argument('-n', '--save-name', default='fcn_l1loss_ffb', help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')

    # Model related
    parser.add_argument('--with_source', default=1, type=int, help='data with source or not')
    parser.add_argument('-m', '--model', default='InversionDeepOnet', type=str, help='inverse model name')
    parser.add_argument('-um', '--up-mode', default=None,
                        help='upsampling layer mode such as "nearest", "bicubic", etc.')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')
    parser.add_argument('--layer_sizes', type=int, default=[5, 256, 256, 256, 512], help='trunk_net layer sizes')
    # Training related
    parser.add_argument('-b', '--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('-lm', '--lr_milestones', nargs='+', default=[], type=int, help='decrease lr on milestones')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')
    parser.add_argument('-ep', '--epoch', type=int, default=120, help='epochs in a saved block')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')

    # Loss related
    parser.add_argument('-g1v', '--lambda_g1v', type=float, default=1)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default=0)
    parser.add_argument('-g3v', '--lambda_g3v', type=float, default=0)
    parser.add_argument('-g4v', '--lambda_perceptual', type=float, default=1)

    # nuaa
    parser.add_argument('--data_url', type=str, default='')  # data_url
    parser.add_argument('--train_model_out', type=str, default='')  # 模型输出路径
    parser.add_argument('--train_out', type=str, default='')  # 文件输出路径
    parser.add_argument('--train_visualized_log', type=str, default='')  # 可视化路径日志
    parser.add_argument('--gpu_num_per_node', type=int, default=1)  # GPU节点个数


    args = parser.parse_args()



    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.log_path = os.path.join(args.log_path, args.save_name, args.suffix or '')
    args.train_anno = os.path.join(args.anno_path, args.train_anno)
    args.val_anno = os.path.join(args.anno_path, args.val_anno)


    if args.resume:
        args.resume = os.path.join(args.output_path, args.resume)

    return args


if __name__ == '__main__':
    args = parse_args()
    start_time = default_timer()
    train(args)
    end_time = default_timer()
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(end_time-start_time)))))

