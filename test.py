from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce
from head import transformer

from config import config
if __name__ == "__main__":
    cfg = config.get_cfg_defaults()
    cfg.freeze()

    head = transformer.Tempr4(  num_freq_bands = cfg.HEAD.NUM_FREQ_BANDS,
                                depth = cfg.HEAD.DEPTH,
                                max_freq = cfg.HEAD.MAX_FREQ,
                                input_channels = cfg.HEAD.INPUT_CHANNELS,
                                input_axis = cfg.HEAD.INPUT_AXIS,
                                num_latents = cfg.HEAD.NUM_LATENTS,
                                latent_dim = cfg.HEAD.LATENT_DIM,
                                cross_heads = cfg.HEAD.CROSS_HEADS,
                                latent_heads = cfg.HEAD.LATENT_HEADS,
                                cross_dim_head = cfg.HEAD.CROSS_DIM_HEAD,
                                latent_dim_head = cfg.HEAD.LATENT_DIM_HEAD,
                                num_classes = cfg.HEAD.NUM_CLASSES,
                                attn_dropout = cfg.HEAD.ATTN_DROPOUT,
                                ff_dropout = cfg.HEAD.FF_DROPOUT,
                                weight_tie_layers = cfg.HEAD.WEIGHT_TIE_LAYERS,
                                fourier_encode_data = cfg.HEAD.FOURIER_ENCODE_DATA,
                                self_per_cross_attn = cfg.HEAD.SELF_PER_CROSS_ATTN,
                                final_classifier_head = cfg.HEAD.FINAL_CLASSIFIER_HEAD)
                
    temp = torch.rand(2, 4, 512, 1, 4, 4)
    result = head(temp)