"""
Dataset : CIFAR-100
Model : ShuffleNetV1
Data Augmentation Technique Used: None
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


# Plotting and Computation Functions

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

    return ECE, OE, avg_acc, avg_conf, sum(acc_counts) / n, counts

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
learning_rate = 0.01     
momentum = 0.9
learning_rate_milestones = [150, 180, 210]
learning_gamma = 0.1
weight_decay = 5e-4

epochs = 500
NUM_BINS = 100



# Model Training Without Data Augmentation
net_simple = ShuffleV1(num_classes=100)
net_simple = net_simple.to(device)
criterion_simple = nn.CrossEntropyLoss().cuda()

train_loader = torch.utils.data.DataLoader(train_d, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_d, shuffle=False, num_workers=8, batch_size=batch_size)

optimiser_simple = torch.optim.SGD(net_simple.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler_simple = lr_scheduler.MultiStepLR(optimiser_simple, milestones=learning_rate_milestones, gamma=learning_gamma)

checkpoint = os.path.join('2Aug/checkpoints/sv1/without')
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
    net_simple.train()
    progress = tqdm(enumerate(train_loader), desc="Epoch: {}".format(epoch), total=len(train_loader))
    for iter, data in progress:
        inputs, targets = data[0], data[1]
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net_simple(inputs)
        loss = criterion_simple(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        optimiser_simple.zero_grad()
        loss.backward()
        optimiser_simple.step()
        progress.update(1)
    scheduler_simple.step()

    print("\nTrain_loss : ",losses.avg)
    ece, oe, bin_acc, bin_conf, acc, bin_count = compute_calibration_metrics(num_bins=NUM_BINS, net=net_simple, loader=test_loader)

    print('Accuracy: {}'.format(acc))
    print('ECE: {}'.format(ece))
    print('OE: {}'.format(oe))

    if (acc > best_acc):
      best_acc = acc
      best_epoch = epoch
      best_ece = ece
      best_oe = oe
      state1 = {
        'state_dict': net_simple.state_dict(),
        'optimizer': optimiser_simple.state_dict(),
        'net': net_simple,
        'acc': best_acc,
        'ece': ece,
        'oe': oe,
        'epoch': best_epoch,
        'rng_state': torch.get_rng_state() 
        }
      print("Best Accuracy checkpoint\n")


    if (epoch == epochs-1):
      state2 = {
        'state_dict': net_simple.state_dict(),
        'optimizer': optimiser_simple.state_dict(),
        'net': net_simple,
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
print("\n")