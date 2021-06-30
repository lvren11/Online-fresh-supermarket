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
from torch.optim import lr_scheduler
from models import *
import vsum_tools
import torch.nn as nn


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_gpu = torch.cuda.is_available()
input_dim = 1000
key = 'video_12'
frame = 1800 #frame<1代表摘要百分比、>1就是帧数,分钟数等于frame/帧数
video_name = 'img_0524_downSample.h5'
resume="D:\\GraduationProject\\pytorch-vsumm-reinforce-master\\pytorch-vsumm-reinforce-copy\\log\\summe-split0\\tvsum_model_epoch_60_split_id_0_feature_1000.pth.tar"
vidataset = 'eccv16_dataset_tvsum_google_pool5(feature1000).h5'
model = MRN(in_dim=input_dim, hid_dim=256, num_layers=1, cell='lstm')
print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
step_size = 30
save_results = 'results/'+video_name
optimizer = torch.optim.Adam(model.parameters(), lr=1e-05, weight_decay=1e-05)
if step_size > 0:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
if use_gpu:
    model = nn.DataParallel(model).cuda()
    #meta_model = nn.DataParallel(meta_model).cuda()
if resume:
    print("Loading checkpoint from '{}'".format(resume))
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint)
else:
    start_epoch = 0

if use_gpu:
    model = nn.DataParallel(model).cuda()

print("==> Test")
with torch.no_grad():
    model.eval()
    fms = []
    eval_metric = 'avg' #可以试试max

    #if args.verbose: table = [["No.", "Video", "F-score"]]
    
    if save_results:
        h5_res = h5py.File(save_results, 'a')
# =============================================================================
#         分割线
# =============================================================================
    '''
    单独的就用这个dataset
    '''
    dataset = h5py.File("D:\\GraduationProject\\extractFrame\\"+video_name,'r')  
    cps = dataset['change_points'][...]
    num_frames = dataset['n_frames'][()]
    nfps = dataset['n_frame_per_seg'][...].tolist()
    positions = dataset['picks'][...]
    seq = dataset['features'][...]
    seq = torch.from_numpy(seq).unsqueeze(0)
    if use_gpu: seq = seq.cuda()
    probs = model(seq)
    probs = probs.data.cpu().squeeze().numpy()
# =============================================================================
#     分割线
# =============================================================================
    '''
    在sumMe或者tvSum数据集中的用这个dataset
    '''
    
#    dataset = h5py.File("D:/GraduationProject/pytorch-vsumm-reinforce-master/pytorch-vsumm-reinforce-copy/datasets/"+vidataset,'r')  
#    #for key_idx, key in enumerate(test_keys):
#    seq = dataset[key]['features'][...]
#    #print(seq)
#
#    seq = torch.from_numpy(seq).unsqueeze(0)
#    if use_gpu: seq = seq.cuda()
#    probs = model(seq)
#    probs = probs.data.cpu().squeeze().numpy()
#    
#    cps = dataset[key]['change_points'][...]
#    num_frames = dataset[key]['n_frames'][()]
#    nfps = dataset[key]['n_frame_per_seg'][...].tolist()
#    positions = dataset[key]['picks'][...]
# =============================================================================
#   分割线
# =============================================================================
    #user_summary = dataset[key]['user_summary'][...]
#    print("cps:",cps)
#    print("num_frames:",num_frames)
#    print("nfps:",nfps)
#    print("positions:",positions)
    machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions,proportion=frame)
    #fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
    #fms.append(fm)
    #print(probs[:600])
    #if args.verbose:
       # table.append([key_idx+1, key, "{:.1%}".format(fm)])
    
    #if args.save_results:
    h5_res.create_dataset('score', data=probs)
    h5_res.create_dataset('machine_summary', data=machine_summary)
    #h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
    #h5_res.create_dataset(key + '/fm', data=fm)

#if args.verbose:
    #print(tabulate(table))

if save_results: h5_res.close()

#mean_fm = np.mean(fms)
#print("Average F-score {:.1%}".format(mean_fm))


#a = sio.loadmat('D:\GraduationProject\SumMe\GT\Air_Force_One.mat')
#print(a.keys())
#import numpy as np
#c = np.random.random((2, 2))
#print(c)
#from utils import Logger, read_json, write_json, save_checkpoint
#splits = read_json("datasets/summe_splits.json")
##print(splits)
#split = splits[0]
##print(split)
#train_keys = split['train_keys']
#print(train_keys)
#baselines = {key: 0. for key in train_keys}
#print(baselines)
#d = np.array(np.random.rand(1,3,2))
#print(d)
#print("------------")
#d = d.squeeze().nonzero()
#print(d)

#b = np.random.rand(1,2)
#a = np.random.rand(5,2)
#print(a)
#print(b)
#a = np.concatenate((b,a))
#print(a.shape)
#print(a)
#import winsound
#winsound.Beep(3000,1000)
#其中600表示声音大小，1000表示发生时长，1000为1秒
#path = 'D:\\GraduationProject\\SumMe\\videos\\Air_Force_One.mp4'
#vidcap = cv2.VideoCapture(path)
#frames = []
#count =0
#print("正在进入")
#while True:
#        
#    success, image = vidcap.read()
#        
#    #frameId = vidcap.get(1)
#    count +=1
#    
#    #print("进来了吗")
#    if not success:
#        break
#    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    cv2.imshow('img_RGB',img_RGB)
#    frames.append(img_RGB)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#vidcap.release()
#
#cv2.destroyAllWindows()
            #print(img_RGB)
#import numpy as np
#X = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])  # 定义二维数组
#print(X[:,0])  # 取数组X二维数组中每一个的0号下标对应的值 [0 4 8 12]
#print(X[[1,2],:])  # 取数组X一维数组中的第一组全部数值  [0 1 2 3]
#print(X[:,1:3])  #取所有数据的第1列到3-1列数据，从第0列开始计算,结果如下：
#
##dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
#print(pick)
#print(train_keys)
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#use_gpu = torch.cuda.is_available()
#print(use_gpu)
#cap = cv2.VideoCapture(0)
#
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#
#out = cv2.VideoWriter('testwrite.avi',cv2.VideoWriter_fourcc(*'XVID'), 20.0, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
#
#while(cap.isOpened()):
#    ret, frame = cap.read()
#    if ret==True:
#
#        cv2.imshow('frame',frame)
#        out.write(frame)
#
#        if cv2.waitKey(10) & 0xFF == ord('q'):
#            break
#    else:
#        break
#
#cap.release()
#out.release()
#cv2.destroyAllWindows()
