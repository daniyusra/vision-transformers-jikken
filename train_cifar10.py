# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np

import torchvision
import torchvision.transforms as transforms
import torchmetrics
from torchsummary import summary


import os
import argparse
import pandas as pd
import csv
import time
from model_loader import load_model
from utils import load_model_quantize

from models import *
from utils import print_size_of_model, progress_bar, load_pretrained
from randomaug import RandAugment
from models.vit import ViT

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', '-aug', action='store_true', help='use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--warmup', '-w', action = "store_true", help = "trigger for warmup")
parser.add_argument('--warmuptype', default="linear", help = "method for warmup")
parser.add_argument("--warmupepochs", default = "0", type=int, help = "amount of warmup epochs")
parser.add_argument("--normlayer", default = "", help = "Normalization layer used for SwinPool")
parser.add_argument('--mixup', default=0.0, type=float, help="enter value between 1.0 and 0.0 to allow mixup augmentation" )
parser.add_argument("--cutmix", default =0.0, type = float, help = "enter value between 1.0 and 0.0 to allow cutmix augmentation")

args = parser.parse_args()

assert args.mixup + args.cutmix <= 1.0

# take in args
usewandb = False
if usewandb:
    import wandb
    watermark = "{}_lr{}".format(args.net, args.lr)
    wandb.init(project="cifar10-challange",
            name=watermark)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = bool(~args.noamp)
aug = args.aug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    print("AUGMENT YO!")
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

quantize = False;

# Model factory..
print('==> Building model..')
# net = VGG('VGG19')

net = load_model(args)

# For Multi-GPU
if 'cuda' in device:
    print(device)
    print("using data parallel")
    net = torch.nn.DataParallel(net) # make parallel
    
    cudnn.benchmark = True

print(summary(net.to(device),(3,size, size)))

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
elif args.opt == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr = args.lr, weight_decay=0.05)
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

#warmup
if (args.warmup):
    scheduler1 = scheduler
    if(args.warmuptype == "linear"):
        scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters= args.warmupepochs)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1,scheduler2])


##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    #region data augmentation
    metric = torchmetrics.Accuracy(task = "multiclass", num_classes=10).to(device)
    def mixup(data, targets, alpha):
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]

        lam = np.random.beta(alpha, alpha)
        new_data = data * lam + shuffled_data * (1 - lam)
        new_targets = [targets, shuffled_targets, lam]
        return new_data, new_targets


    def mixup_criterion(preds, targets):
        targets1, targets2, lam = targets[0], targets[1], targets[2]
        mixupcriterion = criterion
        return lam * mixupcriterion(preds, targets1) + (1 - lam) * mixupcriterion(preds, targets2)

    def mixup_accuracy(metric, preds, y_a, y_b, lam):
        return lam * metric(preds, y_a) + (1 - lam) * metric(preds, y_b)

    def cutmix(data, targets, alpha):
        def rand_bbox(size, lam):
            W = size[2]
            H = size[3]
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            # uniform
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            return bbx1, bby1, bbx2, bby2
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]

        lam = np.random.beta(alpha, alpha)
        bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
        data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

        new_targets = [targets, shuffled_targets, lam]
        return data, new_targets

    #endregion


    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp

        p = np.random.rand()
        if p < args.mixup:
            inputs, targets = mixup(inputs, targets, 0.8)
        elif p < args.cutmix + args.mixup:
            inputs, targets = cutmix(inputs, targets, 0.8)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            if p < args.mixup:
                loss = mixup_criterion(outputs, targets)
                total += outputs.size(0)
                correct += mixup_accuracy(metric, outputs, targets[0],targets[1], targets[2])
            elif p < args.cutmix + args.mixup:
                loss = mixup_criterion(outputs, targets)
                total += outputs.size(0)
                correct += mixup_accuracy(metric, outputs, targets[0],targets[1], targets[2])
            else:
                loss = criterion(outputs, targets) 
                total += targets.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        #total += targets.size(0)
        #correct += predicted.eq(targets).sum().item()

        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100.*correct/total

    return train_loss/(batch_idx+1), acc

##### Validation
def test(net, epoch):
    global best_acc
    start = time.time()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    elapsedTime = time.time() - start

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc, elapsedTime

list_loss = []
list_acc = []
list_loss_train = []
list_acc_train = []
list_train_time = []
list_test_time = []

if usewandb:
    wandb.watch(net)
    
net.cuda()
if (args.warmup):
    for epoch in range(args.warmupepochs):
        print(scheduler.get_last_lr())
        start = time.time()
        trainloss, trainacc = train(epoch)
        list_train_time.append(time.time()-start)

        
        val_loss, acc, start = test(net, epoch)
        list_test_time.append(start)

        scheduler.step()
        
        list_loss_train.append(trainloss)
        list_acc_train.append(trainacc)
        list_loss.append(val_loss)
        list_acc.append(acc)
        
        # Log training..
        if usewandb:
            wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time()-start})

        # Write out csv..
        with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss_train) 
            writer.writerow(list_acc_train)  
            writer.writerow(list_loss) 
            writer.writerow(list_acc)
            writer.writerow(list_train_time) 
            writer.writerow(list_test_time)
            

            
        print(list_loss)
    print("warmup phase complete")

for epoch in range(start_epoch+args.warmupepochs, args.n_epochs+args.warmupepochs):
    print(scheduler.get_last_lr())
    start = time.time()
    trainloss, trainacc = train(epoch)
    list_train_time.append(time.time()-start)

    
    val_loss, acc, start = test(net, epoch)
    list_test_time.append(start)

    #scheduler.step() # step cosine scheduling
    scheduler.step()
    
    list_loss_train.append(trainloss)
    list_acc_train.append(trainacc)
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss_train) 
        writer.writerow(list_acc_train)  
        writer.writerow(list_loss) 
        writer.writerow(list_acc)
        writer.writerow(list_train_time) 
        writer.writerow(list_test_time)
         

        
    print(list_loss)


if quantize:
    load_model_quantize(qnet, net.module)
 
    qnet.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(qnet, inplace=True)
    qnet.cuda()
    test(qnet, 201)
    print("jancok")
    qnet = qnet.to('cpu')
    qnet.eval() 
    torch.quantization.convert(qnet, inplace=True)
    print("QNET model")
    print_size_of_model(qnet)

    print("NET MODEL")
    print_size_of_model(net)
    torch.set_num_threads(1) 
    #qnet = qnet.to('cpu')
    device = 'cpu'
    test(qnet, 201)
    
    
    

# writeout wandb
if usewandb:
    wandb.save("wandb_{}.h5".format(args.net))
    
