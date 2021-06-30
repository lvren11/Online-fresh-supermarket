# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:00:22 2020

@author: 86139
"""
import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
from model_trans import *
from utils import Logger, read_json, write_json, save_checkpoint
from models import *
from rewards import compute_reward
import vsum_tools
from torch.autograd import Variable
import torch.utils.data as Data
import re
import random
#splits = read_json('datasets/summe_splits.json')
#split = splits[0]
#train_keys = split['train_keys']
#test_keys = split['test_keys']
#for i in range(len(train_keys)):
#    train_keys[i] = int(re.sub("\D","",train_keys[i]))
#for i in range(len(test_keys)):    
#    test_keys[i] = int(re.sub("\D","",test_keys[i]))
#train = torch.Tensor(train_keys).unsqueeze(-1)
#test = torch.Tensor(test_keys).unsqueeze(-1)
y = torch.Tensor([])
y=y.squeeze()
print('numel:',y.numel())
print(y.size())
print(y.size(0))
#bsz = 5
#train_dataset = torch_dataset = Data.TensorDataset(train)
#loader = Data.DataLoader(
#        dataset = train_dataset,
#        batch_size = bsz,
#        shuffle = True,
#)
#for epoch in range(5):
#    i = 0
#    choose_list = random.sample(train_keys,bsz)
#    print(choose_list)
#    train_keys = list(set(train_keys)-set(choose_list))
#    if train_keys == []:
#        break
#    for batch_x in loader:
#        for k in batch_x:
#            print(torch.take(k,))
#        #print('epoch:{} | num:{} | batch_x:{}'.format(epoch,i,batch_x))
#        break
#
#x = torch.linspace(min(train_keys),max(train_keys),len(train_keys))
#print(x)
#device = 'cpu'
#def meta_task_data(seed = 0, a_range=[0.1, 5], b_range = [0, 2*np.pi], task_num = 100,
#                   n_sample = 10, sample_range = [-5, 5], plot = False):
#    np.random.seed = seed
#    a_s = np.random.uniform(low = a_range[0], high = a_range[1], size = task_num)
#    b_s = np.random.uniform(low = b_range[0], high = b_range[1], size = task_num)
#    total_x = []
#    total_y = []
#    label = []
#    for t in range(task_num):
#        x = np.random.uniform(low = sample_range[0], high = sample_range[1], size = n_sample)
#        total_x.append(x)
#        total_y.append( a_s[t]*np.sin(x+b_s[t]) )
#        label.append('{:.3}*sin(x+{:.3})'.format(a_s[t], b_s[t]))
#    if plot:
#        plot_x = [np.linspace(-5, 5, 1000)]
#        plot_y = []
#        for t in range(task_num):
#            plot_y.append( a_s[t]*np.sin(plot_x+b_s[t]) ) 
#        return total_x, total_y, plot_x, plot_y, label
#    else:
#        return total_x, total_y, label
#
#bsz = 1
#train_x, train_y, train_label = meta_task_data() 

#train_x = torch.Tensor(train_x).unsqueeze(-1) # add one dim
#train_y = torch.Tensor(train_y).unsqueeze(-1)
#train_dataset = data.TensorDataset(train_x, train_y)
#train_loader = data.DataLoader(dataset=train_dataset, batch_size=bsz, shuffle=False)
##
#test_x, test_y, plot_x, plot_y, test_label = meta_task_data(task_num=1, n_sample = 10, plot=True)  
#print("test_x:",test_x)
#print("_______________")
#print("test_y:",test_y)
#print("_______________")
#print("test:",test_label)
#test_x = torch.Tensor(test_x).unsqueeze(-1) # add one dim
#test_y = torch.Tensor(test_y).unsqueeze(-1) # add one dim
#plot_x = torch.Tensor(plot_x).unsqueeze(-1) # add one dim
#test_dataset = data.TensorDataset(test_x, test_y)
#test_loader = data.DataLoader(dataset=test_dataset, batch_size=bsz, shuffle=False)  
