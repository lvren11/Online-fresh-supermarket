# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 17:42:30 2020

@author: 86139
"""
import numpy as np
import cv2
import os
import torch
import scipy.io as sio
import h5py
h5_res = h5py.File("D:/GraduationProject/pytorch-vsumm-reinforce-master/pytorch-vsumm-reinforce-copy/datasets/eccv16_dataset_summe_google_pool5.h5",'r')
summary = h5_res.keys()
for key in h5_res.keys(): 
    print(key,end=':')
    print(h5_res[key]['video_name'][()])