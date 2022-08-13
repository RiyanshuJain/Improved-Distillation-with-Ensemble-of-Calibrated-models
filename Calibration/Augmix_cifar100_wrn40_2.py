"""
Dataset : CIFAR-100
Model : Wide Resnet 40-2
Data Augmentation Technique Used: AugMix
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
from functools import partial
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

IMAGE_SIZE = 32


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

mixture_depth = -1
aug_severity = 3
mixture_width = 3

def aug(image, preprocess):
  ws = np.float32(np.random.dirichlet([1] * mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(mixture_width):
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(augmentations)
      image_aug = op(image_aug, aug_severity)
    mix += ws[i] * preprocess(image_aug)
  mixed = (1 - m) * preprocess(image) + m * mix

  return mixed


class AugMixDataset(Dataset):
  def __init__(self, dataset, preprocess, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd
    
  def __getitem__(self, i):
    x, y = self.dataset[i]
    # return aug(x, self.preprocess), y
    im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess))
    return im_tuple, y

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

    return ECE, OE, avg_acc, avg_conf, round(sum(acc_counts) / n, 6), counts

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

transform_train = transforms.Compose(
      [transforms.RandomRotation(15),
       transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4)])
preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize(mean, std_dev)])
transform_test = preprocess

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


# Model Training With AugMix
net_am = wrn_40_2(num_classes = 100)
net_am = net_am.to(device)
criterion_am = nn.CrossEntropyLoss().cuda()

train_d_am = AugMixDataset(train_d, preprocess, no_jsd = False)

train_loader = torch.utils.data.DataLoader(train_d_am, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_d, shuffle=False, num_workers=8, batch_size=batch_size)

optimiser_am = torch.optim.SGD(net_am.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler_am = lr_scheduler.MultiStepLR(optimiser_am, milestones=learning_rate_milestones, gamma=learning_gamma)

checkpoint = os.path.join('2Aug/checkpoints/wrn40_2/augmix')
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
    net_am.train()
    progress = tqdm(enumerate(train_loader), desc="Epoch: {}".format(epoch), total=len(train_loader))

    for iter, data in progress:
      images, targets = data[0], data[1]
      targets = targets.to(device)
      images_all = torch.cat(images, 0).to(device)
      logits_all = net_am(images_all)
      logits_clean, logits_aug1, logits_aug2 = torch.split(
          logits_all, images[0].size(0))

      # Cross-entropy is only computed on clean images
      loss = F.cross_entropy(logits_clean, targets)

      p_clean, p_aug1, p_aug2 = F.softmax(
          logits_clean, dim=1), F.softmax(
              logits_aug1, dim=1), F.softmax(
                  logits_aug2, dim=1)

      # Clamp mixture distribution to avoid exploding KL divergence
      p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
      
      loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

      losses.update(loss.item(), images_all.size(0))

      optimiser_am.zero_grad()
      loss.backward()
      optimiser_am.step()
      progress.update(1)
    scheduler_am.step()
    
    print("\nTrain_loss : ",losses.avg)
    ece, oe, bin_acc, bin_conf, acc, bin_count = compute_calibration_metrics(num_bins=NUM_BINS, net=net_am, loader=test_loader)
    
    print('Accuracy: {}'.format(acc))
    print('ECE: {}'.format(ece))
    print('OE: {}'.format(oe))

    if (acc > best_acc):
      best_acc = acc
      best_epoch = epoch
      best_ece = ece
      best_oe = oe
      state1 = {
        'state_dict': net_am.state_dict(),
        'optimizer': optimiser_am.state_dict(),
        'net': net_am,
        'acc': best_acc,
        'ece': ece,
        'oe': oe,
        'epoch': best_epoch,
        'rng_state': torch.get_rng_state() 
        }
      print("Best Accuracy checkpoint\n")

    if (epoch == epochs-1):
      state2 = {
        'state_dict': net_am.state_dict(),
        'optimizer': optimiser_am.state_dict(),
        'net': net_am,
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