import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NewDiceLoss(nn.Module):
    def __init__(self):
        super(NewDiceLoss, self).__init__()

    def forward(self, input, target, beta=1):
        smooth = 10e-4
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        if torch.max(target) == 0:
            target_flat = 1 - target_flat
            input_flat = 1 - input_flat

        tp = input_flat * target_flat
        fp = input_flat * (1 - target_flat)
        fn = (1 - input_flat) * target_flat
        loss = 1 - ((2 * tp.sum(0) + smooth) / (2 * tp.sum(0) + fp.sum(0) + beta * fn.sum(0) + smooth))
        # print(inter.sum(0), input.sum(0), target.sum(0))

        return loss


def cal_dice_loss(input, target, beta):
    smooth = 10e-4
    # input_flat = input.view(-1)
    # target_flat = target.view(-1)
    #
    # inter = input_flat * target_flat
    # loss = 1 - ((2 * inter.sum(0) + smooth) / (input_flat.sum(0) + target_flat.sum(0) + smooth))
    # # print(inter.sum(0), input.sum(0), target.sum(0))

    tp = np.sum(input * target)
    fp = np.sum(input * (1 - target))
    fn = np.sum((1 - input) * target)
    loss = 1 - (2 * tp + 10e-4) / (2 * tp + fp + beta * fn + 10e-4)

    return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


# version 1: use torch.autograd
class FocalLossV1(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        Usage is like this:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
            >>> loss = criteria(logits, lbs)
        '''

        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss
