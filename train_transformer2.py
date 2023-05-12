from config.config import get_cfg_defaults
from model import generate_model
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat
from data_loader import iterator_factory_augmentResNet
from loss import create_loss
from loss import metric
from datetime import datetime
from torch.optim import SGD, lr_scheduler
import os
import numpy as np

class Combine(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.backbone = generate_model(cfg.BACKBONE)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = generate_model(cfg.HEAD)

        self.file_name = cfg.TRAIN.MODEL_NAME

    def forward(self, data):

        B, *_ = data.shape

        data = rearrange(data, 'b s c t h w -> (b s) c t h w')

        # feature: ((b s), c, t, h, w)
        # pred: ((b s), n)
        features, pred = self.backbone(data)

        features = rearrange(features, '(b s) c t h w -> b s c t h w', b = B)

        # features: (b, s, c, t, h, w)
        outputs = self.head(features)
        return outputs

    def save_model(self, file_path, epoch):
        file_name = f"{file_path}-epoch:{epoch}.pth"
        torch.save(self.state_dict(), file_name)
        pass

    def load_model(self, file_path):
        result = torch.load(file_path)
        pass

def load_checkpoint(cfg ,model, optimizer, learning_rate_scheduler):

    checkpoint = torch.load(cfg.TRAIN.PRETRAIN_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    learning_rate_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']

    return epoch, best_acc

def save_checkpoint(cfg, model, optimizer, learning_rate_scheduler, epoch, best_acc):

    file_path = f"{cfg.TRAIN.MODEL_NAME}-epoch:{epoch}.pth"

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': learning_rate_scheduler.state_dict(),
            'best_acc': best_acc
            }, file_path)

def save_best(cfg, model, optimizer, learning_rate_scheduler, epoch):


    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': learning_rate_scheduler.state_dict()
            }, f'{cfg.TRAIN.RESULT_DIR}/model_best.pth')

def create_file_log(cfg):
    """
    Update file dir, file train log, file val log, model name
    """
    now = datetime.now().strftime("%y_%m_%d")
    file_dir = f"{now}-{cfg.BACKBONE.NAME}-{cfg.HEAD.NAME}-video_per:{cfg.DATA.VIDEO_PER}-num_samplers:{cfg.DATA.NUM_SAMPLERS}-optimize:{cfg.TRAIN.OPTIMIZER}-loss:{cfg.TRAIN.LOSS}"
    file_dir = os.path.join(cfg.TRAIN.RESULT_DIR, file_dir)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    cfg.TRAIN.RESULT_DIR = file_dir
    cfg.TRAIN.LOG_FILE_TRAIN = file_dir + '/train.csv'
    cfg.TRAIN.LOG_FILE_VAL = file_dir + '/val.csv'
    cfg.TRAIN.MODEL_NAME = file_dir + f"/{cfg.BACKBONE.NAME}-{cfg.HEAD.NAME}-video_per:{cfg.DATA.VIDEO_PER}-num_samplers:{cfg.DATA.NUM_SAMPLERS}-optimize:{cfg.TRAIN.OPTIMIZER}-loss:{cfg.TRAIN.LOSS}"
    
    # save config file
    with open(file_dir + '/config.yaml', 'w') as f:
        f.write(cfg.dump())
    pass

def get_module_name(name):
    name = name.split('.')
    if name[0] == 'backbone':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]

def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True

        if add_flag:
            parameters.append({'params': v})
        add_flag = False

    return parameters

def create_optimizer(model, cfg):

    if cfg.TRAIN.OPTIMIZER == 'SGD':

        if cfg.BACKBONE.PRETRAINED_MODEL != '':
            parameters = get_fine_tuning_parameters(model, 'head')
        else:
            parameters = model.parameters()

        # parameters = model.parameters()

        optimizer = SGD(parameters,
                        lr=cfg.TRAIN.LEARNING_RATE,
                        momentum=cfg.TRAIN.MOMENTUM,
                        dampening=cfg.TRAIN.DAMPENING,
                        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                        nesterov=cfg.TRAIN.NESTEROV)
    if cfg.TRAIN.OPTIMIZER == 'adam':
    
        params = {
            'classifier':{'lr':cfg.TRAIN.LR_MULT[2],
                    'params':[]},
            'head':{'lr':cfg.TRAIN.LR_MULT[0],
                    'params':[]},
            'pool':{'lr':cfg.TRAIN.LR_MULT[1],
                    'params':[]},
            'base':{'lr':cfg.TRAIN.LEARNING_RATE,
                    'params':[]}
        }

        # Iterate over all parameters
        for name, param in model.named_parameters():
            if 'fc' in name.lower():
                params['classifier']['params'].append(param)
            elif 'head' in name.lower():
                params['head']['params'].append(param)
            elif 'pred_fusion' in name.lower():
                params['pool']['params'].append(param)
            params['base']['params'].append(param)

        optimizer = torch.optim.Adam([
                {'params': params['classifier']['params'], 'lr_mult': params['classifier']['lr']},
                {'params': params['head']['params'], 'lr_mult': params['head']['lr']},
                {'params': params['pool']['params'], 'lr_mult': params['pool']['lr']},],
                lr = cfg.TRAIN.LEARNING_RATE,
                weight_decay = cfg.TRAIN.WEIGHT_DECAY)
    

    return optimizer

def create_lr_scheduler(cfg, optimizer):
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                        cfg.TRAIN.LR_STEP)
    
    return scheduler

def adjust_learning_rate(lr, optimiser):
        # learning rate adjustment based on provided lr rate
        for param_group in optimiser.param_groups:
            if 'lr_mult' in param_group:
                lr_mult = param_group['lr_mult']
            else:
                lr_mult = 1.0
            param_group['lr'] = lr * lr_mult

if __name__ == '__main__':
    cfg = get_cfg_defaults()

    # merge with .yaml

    # input is the config in log dicrectory of model pretrain
    if cfg.TRAIN.TRAIN_CHECKPOINT == True:
        cfg.merge_from_file(cfg.TRAIN.PRETRAIN_CONFIG)
        cfg.TRAIN.TRAIN_CHECKPOINT = True
    else:
        # create file log path
        cfg.merge_from_file('config/perceiver2.yaml')
        create_file_log(cfg)
    

    # cfg.DATA.NUM_SAMPLERS = int(187/(cfg.DATA.FRAME_SKIP + cfg.DATA.CLIP_LENGTH)) + 1
    # cfg.HEAD.DEPTH = cfg.DATA.NUM_SAMPLERS

    cfg.freeze()

    model = Combine(cfg).to(cfg.TRAIN.DEVICE)
    # model = Combine(cfg).to('cpu')

    # create data loader
    if not cfg.DATA.EVAL_ONLY:
        train_loader, val_loader, len_video_train, len_video_val = iterator_factory_augmentResNet.create(cfg.DATA)
        train_metric = metric.MyMetric(cfg, file_path = cfg.TRAIN.LOG_FILE_TRAIN, num_iter = len(train_loader))
        val_metric = metric.MyMetric(cfg, file_path = cfg.TRAIN.LOG_FILE_VAL, num_iter = len(val_loader))
    else:
        val_loader, len_video_val = iterator_factory_augmentResNet.create(cfg.DATA)
        val_metric = metric.MyMetric(cfg, file_path = cfg.TRAIN.LOG_FILE_VAL, num_iter = len(val_loader))

    criterion = create_loss.CrossEntropyLossMean
    optim = create_optimizer(model, cfg)
    learning_rate_scheduler = create_lr_scheduler(cfg, optim)
 
    start_epoch = 0
    best_acc = 0
    if cfg.TRAIN.TRAIN_CHECKPOINT == True:
        start_epoch, best_acc = load_checkpoint(cfg, model, optim, learning_rate_scheduler)
        start_epoch += 1
        pass

    # ---------- Code train AI --------------
    if not cfg.DATA.EVAL_ONLY:
        # ---------- Code train AI --------------
        for epoch in range(start_epoch, cfg.TRAIN.EPOCH):

            model.train()
            train_metric.reset_epoch()
            val_metric.reset_epoch()
            # train
            for i, (data, targets, video_path) in enumerate(train_loader):
                # import numpy as np
                # video = (data[0, 0, ...]*255).detach().cpu().numpy().transpose(1,2,3,0).astype(np.uint8)
                # xxx = video[0]
                # reset metric
                train_metric.reset_batch()

                # data shape (B, S, C, T, H, W)
                data = data.cuda()
                targets = targets.cuda()

                # zero the parameter gradients
                # data = rearrange(data, 'b s c t h w -> (b s) c t h w')
                outputs = model(data)
                # outputs = rearrange(outputs, '(b s) c -> b s c', s = cfg.DATA.NUM_SAMPLERS)
                # loss = 0
                loss = criterion(outputs, targets)

                # outputs = repeat(outputs, 'b n -> b s n', s = cfg.DATA.NUM_SAMPLERS)

                optim.zero_grad()
                loss.backward()
                # temp here, we will apply multi grid training
                # adjust_learning_rate(0.01, optim)
                optim.step()

                # calculate metric in here
                train_metric.update(outputs, targets, loss)
                # print log
                if i != len(train_loader) - 1:
                    train_metric.logg(i, epoch = epoch)
                else:
                    train_metric.logg(i, True, epoch)

            
            learning_rate_scheduler.step()
            # end epoch
            if epoch%cfg.TRAIN.SAVE_FREQUENCY == 0 or epoch == cfg.TRAIN.EPOCH - 1:
                save_checkpoint(cfg, model, optim, learning_rate_scheduler, epoch, best_acc)
            # write in file
            train_metric.write_file(epoch)

            # evaluation
            model.eval()
            with torch.no_grad():

                accuracy_val = 0
                loss_val = 0

                for i, (data, targets, video_path) in enumerate(val_loader):
                    val_metric.reset_batch()
                    if i == len(val_loader) - 1:
                        hihi = 0
                    data = data.cuda()
                    targets = targets.cuda()
                    # zero the parameter gradients
                    outputs = model(data)
                    # outputs = rearrange(outputs, '(b s) c -> b s c', s = cfg.DATA.NUM_SAMPLERS)
                    loss = criterion(outputs, targets)
                    
                    # outputs = repeat(outputs, 'b n -> b s n', s = cfg.DATA.NUM_SAMPLERS)

                    # calculate metric here
                    val_metric.update(outputs, targets, loss)

                    # print log
                    if i != len(val_loader) - 1:
                        val_metric.logg(i, epoch = epoch)
                    else:
                        val_metric.logg(i, True, epoch)


                # write in file
                val_metric.write_file(epoch)

                if val_metric.best_acc >= best_acc:
                    save_best(cfg, model, optim, learning_rate_scheduler, epoch)
    else:
        model.eval()
        # evaluation
        with torch.no_grad():

            accuracy_val = 0
            loss_val = 0

            for i, (data, targets, video_path) in enumerate(val_loader):
                val_metric.reset_batch()
                if i == len(val_loader) - 1:
                    hihi = 0
                data = data.cuda()
                targets = targets.cuda()

                video = (data[0, 0, ...]*255).detach().cpu().numpy().transpose(1,2,3,0).astype(np.uint8)
                xxx = video[0]
                # zero the parameter gradients
                outputs = model(data)
                # outputs = rearrange(outputs, '(b s) c -> b s c', s = cfg.DATA.NUM_SAMPLERS)
                loss = criterion(outputs, targets)

                # calculate metric here
                val_metric.update(outputs, targets, loss)

                # print log
                if i != len(val_loader) - 1:
                    val_metric.logg(i, epoch = 0)
                else:
                    val_metric.logg(i, True, 0)


            # write in file
            val_metric.write_file(0)


        
