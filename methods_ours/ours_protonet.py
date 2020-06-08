# This code is modified from https://github.com/jakesnell/prototypical-networks

import torch
import torch.nn as nn
import numpy as np
from methods_ours.meta_template import MetaTemplate
from torch.autograd import Variable
from tensorboardX import SummaryWriter


class Ours_protonet(MetaTemplate):
    def __init__(self,model_func, n_way, n_support,domain_specific,fine_tune,train_lr,tf_path=None):
        super(Ours_protonet, self).__init__(model_func, n_way, n_support,
                                            domain_specific=domain_specific,fine_tune=fine_tune,train_lr=train_lr, tf_path=None)

    def set_forward(self, x, is_feature=False):
        _, z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


