import os
import argparse

import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from datasets import CIFAR10_Redefined, FashionMNIST_Redefined, MNIST_Redefined
from geometry_v2 import batch_apply_transformation_and_get_label, apply_transformation_with_order
from metric import metric
from models.wide_resnet import WideResNet


N_TRANSFORMATION = 72


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def getCifar10Transform(is_training):
    if is_training:
        transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transform


def getFMNISTTransform(is_training):
    if is_training:
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    return transform


def main(args):
    if args.dataset == 'cifar10':
        train_dataset = CIFAR10_Redefined(in_class=args.in_class, root=args.dataset_path,
                                          is_training=True, transform=getCifar10Transform(True))
        test_dataset = CIFAR10_Redefined(in_class=args.in_class, root=args.dataset_path,
                                         is_training=False, transform=getCifar10Transform(False))
        image_channels = 3
    elif args.dataset == 'fmnist':
        train_dataset = FashionMNIST_Redefined(in_class=args.in_class, root=args.dataset_path, is_training=True,
                                               transform=getFMNISTTransform(True))
        test_dataset = FashionMNIST_Redefined(in_class=args.in_class, root=args.dataset_path, is_training=False,
                                              transform=getFMNISTTransform(False))
        image_channels = 1
    elif args.dataset == 'mnist':
        train_dataset = MNIST_Redefined(in_class=args.in_class, root=args.dataset_path, is_training=True,
                                        transform=getFMNISTTransform(True))
        test_dataset = MNIST_Redefined(in_class=args.in_class, root=args.dataset_path, is_training=False,
                                        transform=getFMNISTTransform(False))
        image_channels = 1
    else:
        raise NotImplementedError

    model = WideResNet(image_channels, num_classes=72, depth=args.wrn_depth, widen_factor=args.wrn_width)
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu))).cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), args.learning_rate,
        weight_decay=args.weight_decay, amsgrad=True
    )
    """
    optimizer = torch.optim.SGD(
        model.parameters(), args.learning_rate, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True)
    """

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=int(args.batch_size / N_TRANSFORMATION), shuffle=True,
        num_workers=args.prefetch, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            (args.epochs + 1) * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-2 / args.learning_rate))

    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, scheduler)

        if epoch != 0 and epoch % args.test_period == 0:
            test(test_loader, model, epoch, args)

    if not (epoch != 0 and epoch % args.test_period == 0):
        test(test_loader, model, epoch, args)


def train(train_loader, model, criterion, optimizer, scheduler):
    model.train()

    for images, _ in train_loader:
        images, k = batch_apply_transformation_and_get_label(images)
        logits = model(images)
        loss = criterion(logits, k)
        print(loss, optimizer.param_groups[0]['lr'])

        scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(test_loader, model, epoch, args):
    model.eval()

    normal_writer_path = os.path.join(args.result_dir, 'normal_score.txt')
    anomaly_writer_path = os.path.join(args.result_dir, 'anomaly_score.txt')
    normal_writer = open(normal_writer_path, 'w')
    anomaly_writer = open(anomaly_writer_path, 'w')

    with torch.no_grad():
        for images_, y in test_loader:
            images = apply_transformation_with_order(images_)
            logits = model(images)
            softmaxs = torch.nn.functional.softmax(logits/args.temperature, dim=1)

            y_ = torch.split(softmaxs, N_TRANSFORMATION, dim=0)

            for i in range(int(images_.size(0))):
                y__ = y_[i].cpu().numpy()
                normality_score = get_normality_score(y__)

                if y[i] == 1:
                    normal_writer.write("{}, {}\n".format(normality_score, 0.))
                else:
                    anomaly_writer.write("{}, {}\n".format(normality_score, 0.))

    metric_dic = metric(normal_writer_path, anomaly_writer_path, epoch, True)
    json_dump = json.dumps(metric_dic)
    f = open(os.path.join(args.result_dir, 'metric.json'), "w")
    f.write(json_dump)
    f.close()


def get_normality_score(softmaxs):
    return np.mean(np.diag(softmaxs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains a one-class model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_class', '-in', type=int, default=0, help='Class to have as the target/in distribution.')
    parser.add_argument('--dataset', '-data', type=str, default='fmnist', help='')
    parser.add_argument('--test_period', '-tp', type=int, default=10, help='')
    parser.add_argument('--result_dir', '-rd', type=str, default='results/', help='')
    parser.add_argument('--dataset_path', '-dp', type=str, default='./data', help='')
    # WideResNet
    parser.add_argument('--wrn_depth', '-wrn_d', type=int, default=16, help='')
    parser.add_argument('--wrn_width', '-wrn_w', type=int, default=4, help='')
    # Temperature Scaling
    parser.add_argument('--temperature', '-t', type=float, default=1., help='')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-4, help='The initial learning rate.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    """
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./snapshots/',
                        help='Folder to save checkpoints.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    """
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=32, help='Pre-fetching threads.')
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    main(args)
