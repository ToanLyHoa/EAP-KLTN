import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat

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

if __name__ == "__main__":

    data = torch.rand(3, 2, 4)
    target = torch.tensor([1, 2, 3])

    print(CrossEntropyLoss(data, target))
    pass
