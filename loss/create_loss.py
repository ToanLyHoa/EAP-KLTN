import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat, reduce
import numpy as np

# def totally_linear(y_true, y_pred):
#         exp_loss = 0
#         T = 18
#         for t in range(1,21):
#                 exp_loss = exp_loss + ((np.double(t)/(T)) * (K.categorical_crossentropy(y_pred, y_true)))

#         return exp_loss


# def totally_expontial(y_true, y_pred):
#     exp_loss = 0
#     T = 18
#     for t in range(0, 21):
#         exp_loss = exp_loss + (np.exp((-1) * (T - t)) * K.categorical_crossentropy(y_pred, y_true))

#     return exp_loss


# def partially_linear(true_dist, coding_dist):
#         loss = 0
#         TIME = 150
#         N_C = 21
#         batch = 32
#         for t in range (TIME):
#                 term1 = true_dist[:,t] * tensor.log(coding_dist[:,t]+0.0000001)
#                 term2 = (1-true_dist[:,t]) * tensor.log(1-coding_dist[:,t]+0.0000001)
#                 loss = loss + np.double(1)/N_C * tensor.sum(term1+term2*np.double(t)/TIME, axis=1)

#         return -loss/batch

def partially_linear(data, target):
    """
    paramaters: 
        data: (b, s, num_classes)
        target: (b, )
    return:
        loss: cross
    """

    batch, TIME, N_C = data.shape
    loss_batch = 0

    target = repeat(target, 'b -> b s', s = TIME)

    for i_batch in range(batch):
        true_dist, coding_dist = data[i_batch],  F.one_hot(target[i_batch], num_classes = 101)
        loss = 0
        for t in range (TIME):
                term1 = true_dist[t, :] * torch.log(coding_dist[t,:]+0.0000001)
                term2 = (1-true_dist[t,:]) * torch.log(1-coding_dist[t,:]+0.0000001)
                loss = loss + np.float64(1)/N_C * torch.sum(term1+term2*np.float64(t)/TIME)
        loss_batch += loss
    return -loss_batch/batch


def CrossEntropyLossMean(data, target):
    """
    paramaters: 
        data: (b, s, num_classes)
        target: (b, )
    return:
        loss: cross
    """

    _, s, _ = data.shape
    data = reduce(data, 'b s n -> b n', 'mean')
    # target = repeat(target, 'b -> (b s)', s = s)
    loss = nn.CrossEntropyLoss().cuda()

    return loss(data, target)

def CrossEntropyLoss(data, target):
    """
    paramaters: 
        data: (b, s, num_classes)
        target: (b, )
    return:
        loss: cross
    """

    _, s, _ = data.shape
    data = rearrange(data, 'b s n -> (b s) n')
    target = repeat(target, 'b -> (b s)', s = s)
    loss = nn.CrossEntropyLoss().cuda()

    return loss(data, target)

def CrossEntropyLossNoScale(data, target):
    """
    paramaters: 
        data: (b, s, num_classes)
        target: (b, )
    return:
        loss: cross
    """

    loss = nn.CrossEntropyLoss().cuda()

    return loss(data, target)

if __name__ == "__main__":

    data = torch.rand(3, 2, 4)
    target = torch.tensor([1, 2, 3])

    print(CrossEntropyLoss(data, target))
    pass
