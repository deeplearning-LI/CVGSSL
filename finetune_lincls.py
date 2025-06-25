import argparse, os, logging, random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from FSDataset import fsdataset

import timm

# ---------------------- Argument Parser ----------------------
parser = argparse.ArgumentParser(description='Finetune & Linear Evaluation')
# Dataset and Model
parser.add_argument('--data', type=str, default='path_to_data')
parser.add_argument('--pretrained', type=str, default='path_to_ckpt.pth.tar')
parser.add_argument('--arch', type=str, default='vit_base_patch16_224')
parser.add_argument('--builder', type=str, default='cvgssl')
# Training Strategy
parser.add_argument('--softmax_lincls', type=bool, default=False)
parser.add_argument('--finetune_model', type=int, default=True)
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--start_epoch', type=int, default=0)
# Experiment Setting
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n', type=int, default=5)  # few-shot runs
parser.add_argument('--shot', type=int, default=5)
parser.add_argument('--class_num', type=int, default=6)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--log_dir', type=str, default='./logs')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()


# ---------------------- Logger Init ----------------------
def init_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    return logging


# ---------------------- Dataset Loader ----------------------
def get_dataloaders(args):
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    test_transform = train_transform

    train_loader_all = []
    for _ in range(args.n):
        train_set = fsdataset(root=os.path.join(args.data, 'train'), factor=args.shot, transform=train_transform)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        train_loader_all.append(train_loader)

    test_set = datasets.ImageFolder(os.path.join(args.data, 'test'), transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return train_loader_all, test_loader


# ---------------------- Model Builder ----------------------
def build_model(args):
    if args.arch.startswith('resnet'):
        model = models.__dict__[args.arch](num_classes=args.class_num).cuda(args.gpu)
        head_name = 'fc'
    else:
        model = timm.create_model(args.arch, num_classes=args.class_num, pretrained=False).cuda(args.gpu)
        head_name = 'head'

    if os.path.isfile(args.pretrained):
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('base_encoder'):
                state_dict[k[len('base_encoder.'):]] = state_dict[k]
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        expected_missing = {f"{head_name}.weight", f"{head_name}.bias"}
        assert set(msg.missing_keys) == expected_missing, f"Unexpected missing keys: {msg.missing_keys}"
    return model


# ---------------------- Train & Validate ----------------------
def train_one_epoch(loader, model, criterion, optimizer, args):
    model.train()
    total_loss, total_correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.cuda(args.gpu), labels.cuda(args.gpu)
        output = model(imgs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += output.argmax(dim=1).eq(labels).sum().item()
    acc = total_correct / len(loader.dataset) * 100
    return total_loss / len(loader), acc


def validate(loader, model, criterion, args):
    model.eval()
    total_loss, total_correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.cuda(args.gpu), labels.cuda(args.gpu)
            output = model(imgs)
            loss = criterion(output, labels)
            total_loss += loss.item()
            total_correct += output.argmax(dim=1).eq(labels).sum().item()
    acc = total_correct / len(loader.dataset) * 100
    return total_loss / len(loader), acc


# ---------------------- Main Function ----------------------
def main(args):
    log_name = f"{args.builder}_finetune_{args.shot}shot_{args.optimizer}_lr{args.lr}.log"
    logging = init_logger(os.path.join(args.log_dir, log_name))
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_loaders, test_loader = get_dataloaders(args)

    for run in range(args.n):
        model = build_model(args)

        if args.softmax_lincls:
            for name, param in model.named_parameters():
                if name not in ['fc.weight', 'fc.bias', 'head.weight', 'head.bias']:
                    param.requires_grad = False
            logging.info("Running in Linear Evaluation Mode")
        elif args.finetune_model:
            logging.info("Running in Finetune Mode")

        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=args.lr)

        best_acc = 0.0
        for epoch in range(args.start_epoch, args.epochs):
            train_loss, train_acc = train_one_epoch(train_loaders[run], model, criterion, optimizer, args)
            val_loss, val_acc = validate(test_loader, model, criterion, args)

            if val_acc > best_acc:
                best_acc = val_acc

            logging.info(f"[Run {run+1}] Epoch [{epoch+1}/{args.epochs}] "
                         f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                         f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        logging.info(f"[Run {run+1}] Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main(args)
