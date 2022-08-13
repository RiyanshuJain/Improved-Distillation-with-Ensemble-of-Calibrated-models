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


# Computation Functions
def compute_calibration_metrics(net1, net2, net3, net4, net5, num_bins=100, loader=None, device='cuda'):
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
    net1.eval()
    net2.eval()
    net3.eval()
    net4.eval()
    net5.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs1 = net1(images, is_feat=False, preact=False)
            outputs2 = net2(images, is_feat=False, preact=False)
            outputs3 = net3(images, is_feat=False, preact=False)
            outputs4 = net4(images, is_feat=False, preact=False)
            outputs5 = net5(images, is_feat=False, preact=False)
            probabilities1 = torch.nn.functional.softmax(outputs1, dim=1)
            probabilities2 = torch.nn.functional.softmax(outputs2, dim=1)
            probabilities3 = torch.nn.functional.softmax(outputs3, dim=1)
            probabilities4 = torch.nn.functional.softmax(outputs4, dim=1)
            probabilities5 = torch.nn.functional.softmax(outputs5, dim=1)
            probabilities = (probabilities1+ probabilities2+ probabilities3 + 1*probabilities4+ 1*probabilities5)/5
            confs, preds = probabilities.max(1)
            #confs2, preds2 = probabilities2.max(1)
            #confs3, preds3 = probabilities3.max(1)
            #confs4, preds4 = probabilities4.max(1)
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
learning_rate = 0.05
momentum = 0.9
learning_rate_milestones = [150, 180, 210]
learning_gamma = 0.1
weight_decay = 5e-4

epochs = 500
NUM_BINS = 100


# Import Teacher model, validate it and define Student model
model_cutmix = wrn_40_2(num_classes = 100)
model = torch.load('2Aug/checkpoints/wrn40_2/cutmix/ckpt-498',map_location=torch.device('cpu'))
model_cutmix.load_state_dict(model['state_dict'])
model_cutmix = model_cutmix.to(device)

for param in model_cutmix.parameters():
    param.requires_grad = False

model_mixup = wrn_40_2(num_classes = 100)
model = torch.load('2Aug/checkpoints/wrn40_2/mixup/ckpt-287',map_location=torch.device('cpu'))
model_mixup.load_state_dict(model['state_dict'])
model_mixup = model_mixup.to(device)

for param in model_mixup.parameters():
    param.requires_grad = False

model_cutout = wrn_40_2(num_classes = 100)
model = torch.load('2Aug/checkpoints/wrn40_2/cutout2/ckpt-370',map_location=torch.device('cpu'))
model_cutout.load_state_dict(model['state_dict'])
model_cutout = model_cutout.to(device)

for param in model_cutout.parameters():
    param.requires_grad = False

model_augmix = wrn_40_2(num_classes = 100)
model = torch.load('2Aug/checkpoints/wrn40_2/augmix/ckpt-204',map_location=torch.device('cpu'))
model_augmix.load_state_dict(model['state_dict'])
model_augmix = model_augmix.to(device)

for param in model_augmix.parameters():
    param.requires_grad = False

model_without = wrn_40_2(num_classes=100)
model = torch.load('2Aug/checkpoints/wrn40_2/without/ckpt-358', map_location=torch.device('cpu'))
model_without.load_state_dict(model['state_dict'])
model_without = model_without.to(device)

for param in model_without.parameters():
    param.requires_grad= False

train_loader = torch.utils.data.DataLoader(train_d, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_d, shuffle=False, num_workers=8, batch_size=batch_size)

print("Validating ensemble model")
ece, oe, bin_acc, bin_conf, acc, bin_count = compute_calibration_metrics(model_mixup, model_cutmix, model_cutout, model_augmix, model_without, 100, loader=test_loader)
print(ece,oe,acc)

