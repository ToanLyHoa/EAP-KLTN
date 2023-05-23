import os
import random
import sys
import coloredlogs, logging
coloredlogs.install()
import math
import torch
import copy
import numpy as np
import imgaug.augmenters as iaa
import torch.multiprocessing as mp
from torch.nn import functional as F
import data_loader.video_transform as transforms
import data_loader.video_iterator as video_iterator 
from data_loader.video_iterator import VideoIter
from torch.utils.data.sampler import RandomSampler
import data_loader.video_sampler as sampler

def get_data(cfg):

    labels_dir = cfg.LABELS_DIR
    eval_only = cfg.EVAL_ONLY
    clip_length = cfg.CLIP_LENGTH
    clip_size = cfg.CLIP_SIZE
    val_clip_length = cfg.VAL_CLIP_LENGTH
    val_clip_size = cfg.VAL_CLIP_SIZE
    val_interval = cfg.VAL_INTERVAL
    mean = cfg.MEAN
    std = cfg.STD
    seed = cfg.SEED
    train_interval = cfg.TRAIN_INTERVAL
    train_file = cfg.TRAIN_FILE
    val_file = cfg.VAL_FILE
    
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    val_sampler = sampler.RandomSequenceFromPoint(  num = clip_length,
                                                    interval = train_interval,
                                                    speed = [1.0, 1.0],
                                                    seed = (seed+0))

    # vid_transform_val=transforms.Compose(
    #     transforms=iaa.Sequential([
    #     iaa.Resize({"shorter-side": clip_size[0], "longer-side":"keep-aspect-ratio"})
    #     ]),
    #     normalise=[mean,std])

    # vid_transform_val=transforms.Compose(normalise=[mean,std])
    vid_transform_val=[]
    

    
    val = VideoIter(csv_filepath=os.path.join(labels_dir, val_file),
                    sampler=val_sampler,
                    video_transform=vid_transform_val,
                    cfg = cfg)
    
    if eval_only:
        return val
    else:
        # Use augmentations only for part of the data
        sometimes_aug = lambda aug: iaa.Sometimes(0.25, aug)
        sometimes_seq = lambda aug: iaa.Sometimes(0.75, aug)
        
        train_sampler = sampler.RandomSequenceFromPoint(num=clip_length,
                                               interval=train_interval,
                                               speed=[1.0, 1.0],
                                               seed=(seed+0))
        
        # vid_transform_train = transforms.Compose(
        #     transforms=iaa.Sequential([
        #     sometimes_seq(iaa.Sequential([
        #     sometimes_aug(iaa.GaussianBlur(sigma=[0.1,0.2])),
        #     sometimes_aug(iaa.Add((-5, 5), per_channel=True)),
        #     sometimes_aug(iaa.AverageBlur(k=(1,2))),
        #     sometimes_aug(iaa.Multiply((0.9, 1.1))),
        #     sometimes_aug(iaa.GammaContrast((0.95,1.05),per_channel=True)),
        #     sometimes_aug(iaa.AddToHueAndSaturation((-7, 7), per_channel=True)),
        #     sometimes_aug(iaa.LinearContrast((0.95, 1.05))),
        #     ]))
        #     ]),
        #     normalise=[mean,std])
        
        vid_transform_train = []


        train = VideoIter(csv_filepath = os.path.join(labels_dir, train_file),
                          sampler = train_sampler,
                          video_transform = vid_transform_train, 
                          cfg = cfg)
        
        # if (val_clip_length == '' and val_clip_size == ''):
        #     return train   

        return train, val


def create(cfg, return_train=True, return_len=False):
    if cfg.EVAL_ONLY:
        val = get_data(cfg)
        val_loader = torch.utils.data.DataLoader(val,
            batch_size = cfg.BATCH_SIZE, shuffle = True,
            num_workers = cfg.WORKERS, pin_memory = False)
        return val_loader, val.__len__()

    dataset_iter = get_data(cfg)
    train, val = dataset_iter
    val_loader = torch.utils.data.DataLoader(val,
        batch_size = cfg.BATCH_SIZE, shuffle = True,
        num_workers = cfg.WORKERS, pin_memory = False)

    train_loader = torch.utils.data.DataLoader(train,
        batch_size = cfg.BATCH_SIZE, shuffle = True,
        num_workers = cfg.WORKERS, pin_memory = False)

    return train_loader, val_loader, train.__len__(), val.__len__()