from config.config import get_cfg_defaults
from model import generate_model
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat
from data_loader import iterator_factory
from loss import create_loss
from loss import metric

class Combine(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.backbone = generate_model(cfg.BACKBONE)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = generate_model(cfg.HEAD)

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

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        pass

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg.freeze()

    model = Combine(cfg).to(cfg.TRAIN.DEVICE)

    train_loader, val_loader, len_video_train, len_video_val = iterator_factory.create(cfg.DATA)


    num_epoch = cfg.TRAIN.EPOCH
    batch_size = cfg.TRAIN.BATCH_SIZE

    criterion = create_loss.CrossEntropyLoss
    optim = torch.optim.Adam(model.parameters(), lr = cfg.TRAIN.LEARNING_RATE)

    train_metric = metric.MyMetric(cfg.DATA.NUM_SAMPLERS, cfg.TRAIN.LOG_FILE_TRAIN,
                                   num_total_data = len(train_loader)*batch_size, 
                                   num_iter= len(train_loader), num_epoch = num_epoch,
                                   batch_size = batch_size)
    
    val_metric = metric.MyMetric(cfg.DATA.NUM_SAMPLERS, cfg.TRAIN.LOG_FILE_VAL,
                                   num_total_data = len(val_loader)*batch_size, 
                                   num_iter= len(val_loader), num_epoch = num_epoch,
                                   batch_size = batch_size)

    for epoch in range(num_epoch):

        # data shape (B, S, C, T, H, W)

        train_metric.reset_epoch()
        val_metric.reset_epoch()
        # train
        for i, (data, targets, video_path) in enumerate(train_loader):
            # reset metric
            train_metric.reset_batch()

            data = data.cuda()
            targets = targets.cuda()

            # zero the parameter gradients
            optim.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optim.step()

            # calculate metric in here
            train_metric.update(outputs, targets, loss)
            # print log
            if i != len(train_loader) - 1:
                train_metric.logg(i)
            else:
                train_metric.logg(i, True, epoch)

        # end epoch

        # write in file
        train_metric.write_file(epoch)

        # evaluation
        with torch.no_grad():

            accuracy_val = 0
            loss_val = 0

            for i, (data, targets, video_path) in enumerate(val_loader):
                
                val_metric.reset_batch()

                # zero the parameter gradients
                outputs = model(data)
                loss = criterion(outputs, targets)

                # calculate metrix here
                val_metric.update(outputs, targets, loss)

                # print log
                if i != len(train_loader) - 1:
                    val_metric.logg(i)
                else:
                    val_metric.logg(i, True, epoch)


            # write in file
            val_metric.write_file(epoch)


        model.save_model(cfg.TRAIN.RESULT_DIR)
