import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.resnet import conv1x1x1, Bottleneck, ResNet
from backbone.utils import partialclass


def get_inplanes():
    return [128, 256, 512, 1024]


class ResNeXtBottleneck(Bottleneck):
    expansion = 2

    def __init__(self, in_planes, planes, cardinality, stride=1,
                 downsample=None):
        super().__init__(in_planes, planes, stride, downsample)

        mid_planes = cardinality * planes // 32
        self.conv1 = conv1x1x1(in_planes, mid_planes)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes,
                               mid_planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               groups=cardinality,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = conv1x1x1(mid_planes, planes * self.expansion)


class ResNeXt(ResNet):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 cardinality=32,
                 n_classes=400,
                 pretrained_path = ''):
        block = partialclass(block, cardinality=cardinality)
        super().__init__(block, layers, block_inplanes, n_input_channels,
                         conv1_t_size, conv1_t_stride, no_max_pool,
                         shortcut_type, 1, n_classes)

        self.fc = nn.Linear(cardinality * 32 * block.expansion, n_classes)
        
        if pretrained_path != '':
            self.load_weight(pretrained_path)
    def load_weight(self, pretrained_path):
        # raw fucntion but OK with it
        pre_trained_model=torch.load(pretrained_path)
        new = list(pre_trained_model['state_dict'].items())

        my_model_kvpair = self.state_dict()
        count=0
        for key,value in my_model_kvpair.items():
            layer_name, weights = new[count]    
            if value.shape == weights.shape:
                my_model_kvpair[key] = weights
            count+=1

        self.load_state_dict(my_model_kvpair)
        # self.load_state_dict(torch.load(pretrained_path)['state_dict'])
        pass

def generate_model(model_depth, **kwargs):
    assert model_depth in [50, 101, 152, 200]

    if model_depth == 50:
        model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], get_inplanes(),
                        **kwargs)
    elif model_depth == 101:
        model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], get_inplanes(),
                        **kwargs)
    elif model_depth == 152:
        model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], get_inplanes(),
                        **kwargs)
    elif model_depth == 200:
        model = ResNeXt(ResNeXtBottleneck, [3, 24, 36, 3], get_inplanes(),
                        **kwargs)

    return model
