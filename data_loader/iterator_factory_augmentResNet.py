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

from data_loader.spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from data_loader.temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)

def get_normalize_method(mean, std):
    return Normalize(mean, std)

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

    normalize = get_normalize_method(mean, std)
    spatial_transform = [
        Resize(cfg.CLIP_SIZE[1]),
        CenterCrop(cfg.CLIP_SIZE[1]),
        ToTensor()
    ]
    value_scale = 1
    spatial_transform.extend([ScaleValue(value_scale), normalize])
    spatial_transform = Compose(spatial_transform)
    

    
    val = VideoIter(csv_filepath=os.path.join(labels_dir, val_file),
                    sampler=val_sampler,
                    video_transform=spatial_transform,
                    cfg = cfg)
    
    if eval_only:
        return val
    else:
        train_sampler = sampler.RandomSequenceFromPoint(num=clip_length,
                                               interval=train_interval,
                                               speed=[1.0, 1.0],
                                               seed=(seed+0))
        train_crop = 'random'
        assert train_crop in ['random', 'corner', 'center']
        spatial_transform = []
        train_crop_min_scale = 0.25
        train_crop_min_ratio = 0.75
        if train_crop == 'random':
            spatial_transform.append(
                RandomResizedCrop(
                    cfg.CLIP_SIZE[1], (train_crop_min_scale, 1.0),
                    (train_crop_min_ratio, 1.0 / train_crop_min_ratio)))
        elif train_crop == 'corner':
            scales = [1.0]
            scale_step = 1 / (2**(1 / 4))
            for _ in range(1, 5):
                scales.append(scales[-1] * scale_step)
            spatial_transform.append(MultiScaleCornerCrop(cfg.CLIP_SIZE[1], scales))
        elif train_crop == 'center':
            spatial_transform.append(Resize(cfg.CLIP_SIZE[1]))
            spatial_transform.append(CenterCrop(cfg.CLIP_SIZE[1]))
        normalize = get_normalize_method(cfg.MEAN, cfg.STD)

        no_hflip = False
        if not no_hflip:
            spatial_transform.append(RandomHorizontalFlip())
        colorjitter = False
        if colorjitter:
            spatial_transform.append(ColorJitter())
        spatial_transform.append(ToTensor())
        spatial_transform.append(ScaleValue(value_scale))
        spatial_transform.append(normalize)
        spatial_transform = Compose(spatial_transform)

        train = VideoIter(csv_filepath = os.path.join(labels_dir, train_file),
                          sampler = train_sampler,
                          video_transform = spatial_transform, 
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