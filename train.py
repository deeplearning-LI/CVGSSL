import argparse
import os
import time
import math
import random
import shutil
import warnings

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

from data_aug import loader, multicrop
from data_aug.loader import GaussianBlur
import builder.cvgssl as cvgssl

import timm
import clip

# -----------------------------------
# Model architecture list
# -----------------------------------
torchvision_model_names = sorted(
    name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)
model_names = ['vit_base_patch16_224', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

# -----------------------------------
# Argument parser
# -----------------------------------
parser = argparse.ArgumentParser(description='CVGSSL Training')
parser.add_argument('--data', default='/NEU/', type=str, help='Path to dataset')
parser.add_argument('--arch', default='resnet18', choices=model_names, help='Model architecture')
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--lr', default=0.3, type=float)
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'adamw'])
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-6, type=float)
parser.add_argument('--warmup-epochs', default=10, type=int)
parser.add_argument('--print-freq', default=10, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--save', default='results', type=str)
parser.add_argument('--save-freq', default=10, type=int)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--out-dim', default=256, type=int)
parser.add_argument('--mlp-dim', default=512, type=int)
parser.add_argument('--t1', default=1.0, type=float)

# Multi-crop arguments
# The crop rate has a significant impact on the model, and it can be set according to the characteristics of the defects.
parser.add_argument('--multi-crop', type=bool, default=True)
parser.add_argument('--nmb-crops', type=int, nargs="+", default=[1, 1, 1, 1])
parser.add_argument('--size-crops', type=int, nargs="+", default=[224, 224, 224, 224])
parser.add_argument('--min-scale-crops', type=float, nargs="+", default=[0.2, 0.172, 0.143, 0.114])
parser.add_argument('--max-scale-crops', type=float, nargs="+", default=[1.0, 0.86, 0.715, 0.571])


def main():
    args = parser.parse_args()

    # Set seed and deterministic behavior
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    if args.arch.startswith('vit_'):
        vit_model = timm.create_model(args.arch, pretrained=False)
        model = cvgssl.CVGSSL_ViT(vit_model, args.out_dim, args.mlp_dim, args.t1)
    else:
        cnn_model = torchvision_models.__dict__[args.arch](zero_init_residual=True)
        model = cvgssl.CVGSSL_ResNet(cnn_model, args.out_dim, args.mlp_dim, args.t1)

    # Download the weight file of CLIP.
    # e.g. : https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
    
    clip_model, _ = clip.load('ckp/**.pt', device=f"cuda:{args.gpu}")
    for p in clip_model.parameters():
        p.requires_grad = False

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler()
    summary_writer = SummaryWriter() if args.gpu == 0 else None

    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']
        print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")

    cudnn.benchmark = True

    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if args.multi_crop:
        transform = multicrop.Multi_Transform(args.size_crops, args.nmb_crops, args.min_scale_crops, args.max_scale_crops, normalize)
        train_dataset = datasets.ImageFolder(traindir, transform)
    else:
        aug1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.55, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.5),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            normalize
        ])
        aug2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.ColorJitter(0.2, 0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        train_dataset = datasets.ImageFolder(traindir, loader.TwoCropsTransform(aug1, aug2))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    log_dir = os.path.join(args.save, f"train-lr{args.lr}-{args.optimizer}")
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(train_loader, model, clip_model, optimizer, scaler, summary_writer, epoch, args)

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(log_dir, f"checkpoint_{epoch:04d}.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=checkpoint_path)
            print("Checkpoint saved.")


def train_one_epoch(train_loader, model, clip_model, optimizer, scaler, summary_writer, epoch, args):
    model.train()
    end = time.time()

    for i, (images, _) in enumerate(train_loader):
        if i == 0:
            print(f"Using {len(images)} multi-crops.")

        images = [img.cuda(args.gpu, non_blocking=True) for img in images]
        lr = adjust_learning_rate(optimizer, epoch + i / len(train_loader), args)

        with torch.cuda.amp.autocast():
            clip_features = [clip_model.encode_image(img).float() for img in images]
            loss = model(images[0], images[1:], clip_features, epoch)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if args.gpu == 0 and summary_writer:
            summary_writer.add_scalar('loss', loss.item(), epoch * len(train_loader) + i)

        if i % args.print_freq == 0:
            print(f"Epoch [{epoch}] Iter [{i}/{len(train_loader)}] Loss: {loss.item():.4f} LR: {lr:.5f}")


def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


if __name__ == '__main__':
    main()
