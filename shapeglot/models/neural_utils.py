"""
The MIT License (MIT)
Originally created sometime in 2019.
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
from torch import nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    def __init__(self, in_feat_dims, out_channels, use_b_norm, dropout,
                 non_linearity=nn.ReLU(inplace=True), closure=None):
        super(MLPDecoder, self).__init__()

        previous_feat_dim = in_feat_dims
        all_ops = []

        for depth in range(len(out_channels)):
            out_dim = out_channels[depth]
            
            if depth == len(out_channels) - 2:   # before middle layer
                if dropout:
                    all_ops.append(nn.Dropout(0.2 if dropout==True else dropout))
                    
            affine_op = nn.Linear(previous_feat_dim, out_dim, bias=True)
            all_ops.append(affine_op)

            if depth < len(out_channels) - 1:   # after first layer and after middle layer
                if use_b_norm:
                    all_ops.append(nn.BatchNorm1d(out_dim))

                if non_linearity is not None:
                    all_ops.append(non_linearity)
            
            previous_feat_dim = out_dim

        if closure is not None:
            all_ops.append(closure)

        self.net = nn.Sequential(*all_ops)

    def forward(self, x):
        return self.net(x)


def smoothed_cross_entropy(pred, target, alpha=0.1):
    n_class = pred.size(1)  # n of samples
    one_hot = torch.zeros_like(pred).scatter_(1, target.view(-1, 1), 1)
    one_hot = one_hot * ((1.0 - alpha) + alpha / n_class) + (1.0 - one_hot) * alpha / n_class  # smoothed
    log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    return torch.mean(loss)

'''
def per_axis_smoothed_cross_entropy(pred, target, alpha=0.1):
    n_class = pred.size(0)  # n of samples
    one_hot = torch.zeros_like(pred).scatter_(1, target.view(-1, 1), 1)
    one_hot = one_hot * ((1.0 - alpha) + alpha / n_class) + (1.0 - one_hot) * alpha / n_class  # smoothed
    log_prb_0 = F.log_softmax(pred, dim=0)
    log_prb_1 = F.log_softmax(pred, dim=1)
    loss0 = -(one_hot * log_prb_0).sum(dim=0)
    loss1 = -(one_hot * log_prb_1).sum(dim=1)
    return torch.mean(loss0), torch.mean(loss1) #torch.mean(loss)
'''