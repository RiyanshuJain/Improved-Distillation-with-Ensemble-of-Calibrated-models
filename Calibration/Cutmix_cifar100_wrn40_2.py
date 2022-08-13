"""
Dataset : CIFAR-100
Model : Wide Resnet 40-2
Data Augmentation Technique Used: CutMix
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, lr_scheduler
from torchvision.datasets import CIFAR100
from torchvision import transforms
from Models import *
from datetime import datetime
import random
from tqdm import tqdm
from PIL import Image
from PIL import ImageOps
import warnings
warnings.filterwarnings('ignore')

torch.cuda.is_available()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TIME_NOW = str(datetime.now().strftime('%Y-%m-%d--%H-%M'))


# Utility Functions
def save_checkpoint(state, path, epoch):
    # Save checkpoint.
    print('Saving..')
    torch.save(state, path+'/ckpt-{}'.format(epoch))
    # torch.save(state, path+'/ckpt-{}.t7'.format(epoch))
    print('Saved model to {}'.format(path))

class CutMixCrossEntropyLoss(Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        return cross_entropy(input, target, self.size_average)


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=1, beta=1., prob=1.0):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)



# Computation Functions
def compute_calibration_metrics(num_bins=100, net=None, loader=None, device='cuda'):
    """
    Computes the calibration metrics ECE and OE along with the acc and conf values
    :param num_bins: Taken from email correspondence and 100 is used
    :param net: trained network
    :param loader: dataloader for the dataset
    :param device: cuda or cpu
    :return: ECE, OE, acc, conf
    """
    acc_counts = [0 for _ in range(num_bins+1)]
    conf_counts = [0 for _ in range(num_bins+1)]
    overall_conf = []
    n = float(len(loader.dataset))
    counts = [0 for i in range(num_bins+1)]
    net.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images, is_feat=False, preact=False)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confs, preds = probabilities.max(1)
            for (conf, pred, label) in zip(confs, preds, labels):
                bin_index = int(((conf * 100) // (100/num_bins)).cpu())
                try:
                    if pred == label:
                        acc_counts[bin_index] += 1.0
                    conf_counts[bin_index] += conf.cpu()
                    counts[bin_index] += 1.0
                    overall_conf.append(conf.cpu())
                except:
                    print(bin_index, conf)
                    raise AssertionError('Bin index out of range!')

    avg_acc = [0 if count == 0 else acc_count / count for acc_count, count in zip(acc_counts, counts)]
    avg_conf = [0 if count == 0 else conf_count / count for conf_count, count in zip(conf_counts, counts)]
    ECE, OE = 0, 0
    for i in range(num_bins):
        ECE += (counts[i] / n) * abs(avg_acc[i] - avg_conf[i])
        OE += (counts[i] / n) * (avg_conf[i] * (max(avg_conf[i] - avg_acc[i], 0)))

    return ECE, OE, avg_acc, avg_conf, round(sum(acc_counts) * 100 / n, 6), counts

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



# Importing The CIFAR-100 Dataset & Defining Hyperparameters
mean = (0.5071, 0.4867, 0.4408)
std_dev = (0.2675, 0.2565, 0.2761)

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std_dev)
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std_dev)
    ])

train_d = CIFAR100(root='./data/', train=True, transform=transform_train, download=True)
test_d = CIFAR100(root='./data/', train=False, transform=transform_test, download=True)

batch_size = 128
learning_rate = 0.05            # for wrn & 0.01 for sv1
momentum = 0.9
learning_rate_milestones = [150, 180, 210]
learning_gamma = 0.1
weight_decay = 5e-4

epochs = 500
NUM_BINS = 100


# Model Training With CutMix
net_cm = wrn_40_2(num_classes = 100)
net_cm = net_cm.to(device)
criterion_cm = CutMixCrossEntropyLoss()

train_d_cm = CutMix(train_d, 100,)

train_loader = torch.utils.data.DataLoader(train_d_cm, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_d, shuffle=False, num_workers=8, batch_size=batch_size)

optimiser_cm = torch.optim.SGD(net_cm.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler_cm = lr_scheduler.MultiStepLR(optimiser_cm, milestones=learning_rate_milestones, gamma=learning_gamma)

checkpoint = os.path.join('2Aug/checkpoints/wrn40_2/cutmix')
log_path = os.path.join(checkpoint, 'logs')
if not os.path.exists(checkpoint):
    os.makedirs(checkpoint)
    os.makedirs(log_path)

writer = SummaryWriter(log_path)

best_acc = 0 
best_epoch = 0
state1 = {}
state2 = {}
losses = AverageMeter()

for epoch in range(epochs):
    net_cm.train()
    progress = tqdm(enumerate(train_loader), desc="Epoch: {}".format(epoch), total=len(train_loader))
    for iter, data in progress:
        inputs, targets = data[0], data[1]
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net_cm(inputs)
        loss = criterion_cm(outputs, targets)
        losses.update(loss.item(), inputs.size(0))

        optimiser_cm.zero_grad()
        loss.backward()
        optimiser_cm.step()
        
        progress.update(1)
    scheduler_cm.step()

    print("\nTrain_loss : ",losses.avg)
    ece, oe, bin_acc, bin_conf, acc, bin_count = compute_calibration_metrics(num_bins=NUM_BINS, net=net_cm, loader=test_loader)
    
    print('Accuracy: {}'.format(acc))
    print('ECE: {}'.format(ece))
    print('OE: {}'.format(oe))
    # print("\n")

    if (acc > best_acc):
      best_acc = acc
      best_epoch = epoch
      best_ece = ece
      best_oe = oe
      state1 = {
        'state_dict': net_cm.state_dict(),
        'optimizer': optimiser_cm.state_dict(),
        'net': net_cm,
        'acc': best_acc,
        'ece': ece,
        'oe': oe,
        'epoch': best_epoch,
        'rng_state': torch.get_rng_state() 
        }
      print("Best Accuracy checkpoint\n")

    if (epoch == epochs-1):
      state2 = {
        'state_dict': net_cm.state_dict(),
        'optimizer': optimiser_cm.state_dict(),
        'net': net_cm,
        'acc': acc,
        'ece': ece,
        'oe': oe,
        'epoch': epoch,
        'rng_state': torch.get_rng_state() 
        }
      save_checkpoint(state2, path=checkpoint, epoch=epoch)
      
save_checkpoint(state1, path=checkpoint, epoch=best_epoch)
print("Best Accuracy --> ", best_acc, end=" ")
print("at epoch --> ", best_epoch)
print("ece achieved --> ", best_ece)
print("oe achieved --> ", best_oe)