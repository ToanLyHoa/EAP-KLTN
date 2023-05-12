# my_project/config.py
from yacs.config import CfgNode as CN
import torch

_C = CN()

_C.BACKBONE = CN()
# name of backbone
_C.BACKBONE.NAME = "resnet3d"
# Depth of resnet (10 | 18 | 34 | 50 | 101)
_C.BACKBONE.MODEL_DEPTH = 18
# class for last fully connected layer
_C.BACKBONE.N_CLASSES = 21
# channel of input
_C.BACKBONE.N_INPUT_CHANNELS = 3
# Shortcut type of resnet (A | B)
_C.BACKBONE.RESNET_SHORTCUT = 'B'
# Kernel size in t dim of conv1.
_C.BACKBONE.CONV1_T_SIZE = 7
# Stride in t dim of conv1.
_C.BACKBONE.CONV1_T_STRIDE = 1
# If true, the max pooling after conv1 is removed.
_C.BACKBONE.NO_MAX_POOL = False
# The number of feature maps of resnet is multiplied by this value
_C.BACKBONE.RESNET_WIDEN_FACTOR = 1.0

# ResNeXt cardinality
_C.BACKBONE.RESNEXT_CARDINALITY = 32

# Pretrained path model
_C.BACKBONE.PRETRAINED_MODEL = '/home/it/Desktop/NTMINH/Khoa_Luan_Tot_Nghiep/Pretrained_backbone/r3d50_K_200ep.pth'

_C.HEAD = CN()
_C.HEAD.NAME = "tempr4"
_C.HEAD.NUM_FREQ_BANDS = 10
_C.HEAD.DEPTH = 3
_C.HEAD.MAX_FREQ = 10.
_C.HEAD.INPUT_CHANNELS = 512
_C.HEAD.INPUT_AXIS = 3
_C.HEAD.NUM_LATENTS = 256
_C.HEAD.LATENT_DIM = 512
_C.HEAD.CROSS_HEADS = 4
_C.HEAD.LATENT_HEADS = 8
_C.HEAD.CROSS_DIM_HEAD = 64
_C.HEAD.LATENT_DIM_HEAD = 64
_C.HEAD.NUM_CLASSES = 21
_C.HEAD.ATTN_DROPOUT = 0
_C.HEAD.FF_DROPOUT = 0
_C.HEAD.WEIGHT_TIE_LAYERS = False
_C.HEAD.FOURIER_ENCODE_DATA = True
_C.HEAD.SELF_PER_CROSS_ATTN = 1
_C.HEAD.FINAL_CLASSIFIER_HEAD = True

_C.TRAIN = CN()

_C.TRAIN.GOOGLE_COLAB = False

_C.TRAIN.TRAIN_CHECKPOINT = False
_C.TRAIN.PRETRAIN_PATH = ''
_C.TRAIN.PRETRAIN_CONFIG = 'log/23_04_14-resnet3d_50--video_per:0.5-num_samplers:1-1-optimize:SGD-loss:crossentropylossmean/config.yaml'

_C.TRAIN.DEVICE = "cuda:0"
_C.TRAIN.EPOCH = 60
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.WORKERS = 1
_C.TRAIN.LEARNING_RATE = 0.01
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.DAMPENING = 0.0
_C.TRAIN.NESTEROV = False
_C.TRAIN.WARM_UP = True

# learning rate multipliers for different sets of parameters. 
# head - pool - classifier
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.LOSS = 'crossentropyloss'
_C.TRAIN.LR_MULT = [1, 1, 1]
_C.TRAIN.LR_STEP = [50, 100, 150]
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.WEIGHT_DECAY = 1.0e-05
_C.TRAIN.SAVE_FREQUENCY = 5
_C.TRAIN.RESULT_DIR = 'log'
_C.TRAIN.LOG_FILE_TRAIN = 'log/train.csv'
_C.TRAIN.LOG_FILE_VAL = 'log/val.csv'
_C.TRAIN.MODEL_NAME = ''


_C.DATA = CN()

_C.DATA.DATA_DIR = 'data/UCF-101-DB'
_C.DATA.TRAIN_FILE = 'new_train.csv'
_C.DATA.VAL_FILE = 'new_val.csv'
_C.DATA.LABELS_DIR = 'label/UCF-101'
_C.DATA.USE_GPU = True if _C.TRAIN.DEVICE != 'cpu' else False
_C.DATA.EVAL_ONLY = False

_C.DATA.VIDEO_PER = .6
_C.DATA.VIDEO_PER_TRAIN = .6
_C.DATA.VIDEO_PER_VAL = .6

_C.DATA.TYPE_SAMPLERS = 'scale' # 'normal' | 'scale' | even_crop'
_C.DATA.FRAME_SKIP = 4

_C.DATA.NUM_SAMPLERS = _C.HEAD.DEPTH
_C.DATA.CLIP_LENGTH = 16
_C.DATA.CLIP_SIZE = _C.DATA.CLIP_LENGTH, 112, 112
_C.DATA.VAL_CLIP_LENGTH = ''
_C.DATA.VAL_CLIP_SIZE = ''

_C.DATA.INCLUDE_TIMESLICES = True
_C.DATA.TRAIN_INTERVAL = [1,2]
_C.DATA.VAL_INTERVAL = [1]

# _C.DATA.MEAN = [0.485, 0.456, 0.406]
# _C.DATA.STD = [0.229, 0.224, 0.225]

_C.DATA.MEAN = [0.4345, 0.4051, 0.3775]
_C.DATA.STD = [0.2768, 0.2713, 0.2737]

_C.DATA.SEED = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
_C.DATA.SHUFFLE_LIST_SEED = _C.DATA.SEED + 2
_C.DATA.RETURN_VIDEO_PATH = True

# GPU-device related parser arguments & number of workers
_C.DATA.WORKERS = _C.TRAIN.WORKERS

# set batch size
_C.DATA.BATCH_SIZE = _C.TRAIN.BATCH_SIZE
_C.DATA.FRAME_SIZE = 224


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`