from yacs.config  import CfgNode as CN
import torch
import os

_C = CN()

# _C.DATA.data_dir = os.path.join('mydata','UCF-101-test')
# _C.DATA.labels_dir = os.path.join('mydata','UCF-101-testcsv')
_C.DATA = CN()

_C.DATA.DATA_DIR = 'UCF-101-test'
_C.DATA.LABELS_DIR = 'UCF-101-testcsv'
_C.DATA.EVAL_ONLY = False

_C.DATA.VIDEO_PER = .6
_C.DATA.VIDEO_PER_TRAIN = .6
_C.DATA.VIDEO_PER_VAL = .6
_C.DATA.NUM_SAMPLERS = 4

_C.DATA.CLIP_LENGTH = 8
_C.DATA.CLIP_SIZE = _C.DATA.CLIP_LENGTH, 224, 224
_C.DATA.VAL_CLIP_LENGTH = ''
_C.DATA.VAL_CLIP_SIZE = ''

_C.DATA.INCLUDE_TIMESLICES = True
_C.DATA.TRAIN_INTERVAL = [1,2]
_C.DATA.VAL_INTERVAL = [1]

_C.DATA.MEAN = [0.485, 0.456, 0.406]
_C.DATA.STD = [0.229, 0.224, 0.225]

_C.DATA.SEED = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
_C.DATA.SHUFFLE_LIST_SEED = _C.DATA.SEED+2
_C.DATA.RETURN_VIDEO_PATH = False

# GPU-device related parser arguments & number of workers
_C.DATA.WORKERS = 2

# set batch size
_C.DATA.BATCH_SIZE = 2
_C.DATA.FRAME_SIZE = 224

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
