import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss
import torch.nn as nn


def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0)) # tensor(13.1001)
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2-arr)**2) # tensor(17.2448)

    return lamda1*loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))


class RTFM_loss(torch.nn.Module): # loss2
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0) # (2*batch_size,)=(32,)
        score_abnormal = score_abnormal # (batch_size,1)=(32,1)
        score_normal = score_normal # (batch_size,1)=(32,1)

        score = torch.cat((score_normal, score_abnormal), 0) # (2*batch_size,1)=(32,1)
        score = score.squeeze() # (32,)

        label = label.cuda()

        loss_cls = self.criterion(score, label)  # BCE loss in the score space ,tensor(0.6887)

        featmean = torch.mean(feat_a, dim=1) 
        featnorm = torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1) #(160,)
        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1)) # (160,3,2048)->(160,2048)->(160,)

        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)

        relu = nn.ReLU(inplace=True)
        test1 = relu(feat_a-feat_n)
        loss_test1 = torch.abs(torch.norm(torch.mean(test1, dim=1), p=2, dim=1))
        loss_test1 = torch.mean(loss_test1)

        loss_maxabn = torch.mean((max(loss_abn) + min(loss_abn)))

        loss_maxnor = torch.mean((max(loss_nor) - min(loss_nor)))

        loss_total = loss_cls + self.alpha * loss_test1 + 0.001 * loss_maxabn + 0.01 * loss_maxnor

        return loss_total

def train(nloader, aloader, model, batch_size, optimizer, viz, device):
    with torch.set_grad_enabled(True):
        model.train() 

        ninput, nlabel = next(nloader) # (bs,10,T=32,2048)
        ainput, alabel = next(aloader) # (bs,10,T=32,2048)

        input = torch.cat((ninput, ainput), 0).to(device) 

        B, N, T, D = input.size()

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
        feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(input)  # b*32  x 2048
        
        scores = scores.view(batch_size * 32 * 2, -1) # (1024,1)

        scores = scores.squeeze() # (1024,)
        abn_scores = scores[batch_size * 32:] # (512,)

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size] 

        loss_criterion = RTFM_loss(0.001, 1)
        loss_sparse = sparsity(abn_scores, batch_size, 8e-3) 
        loss_smooth = smooth(abn_scores, 8e-4)
        cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn) + loss_smooth + loss_sparse

        viz.plot_lines('loss', cost.item())
        viz.plot_lines('smooth loss', loss_smooth.item())
        viz.plot_lines('sparsity loss', loss_sparse.item())
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()