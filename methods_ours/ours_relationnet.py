import torch
import torch.nn as nn
import numpy as np
from methods_ours.meta_template import MetaTemplate
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from methods import backbone
import torch.nn.functional as F
import utils
class ours_RelationNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support,domain_specific,fine_tune,train_lr,tf_path=None, loss_type='mse'):
        super(ours_RelationNet, self).__init__(model_func, n_way, n_support,
        domain_specific=domain_specific,fine_tune=fine_tune,train_lr=train_lr,flatten=False, tf_path=tf_path)

        # loss function
        self.loss_type = loss_type  # 'softmax' or 'mse'
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        # metric function
        self.relation_module = RelationModule(self.feat_dim, 8,
                                              self.loss_type)  # relation net features are not pooled, so self.feat_dim is [dim, w, h]
        self.method = 'RelationNet'
        self.domain_specific=domain_specific

    def set_forward(self, x, is_feature=False):

        # get features
        z_all, z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, *self.feat_dim).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, *self.feat_dim)

        # get relations with metric function
        z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query * self.n_way, 1, 1, 1, 1)
        z_query_ext = z_query.unsqueeze(0).repeat(self.n_way, 1, 1, 1, 1)
        z_query_ext = torch.transpose(z_query_ext, 0, 1)
        extend_final_feat_dim = self.feat_dim.copy()
        extend_final_feat_dim[0] *= 2
        relation_pairs = torch.cat((z_proto_ext, z_query_ext), 2).view(-1, *extend_final_feat_dim)
        relations = self.relation_module(relation_pairs).view(-1, self.n_way)
        return relations

    def domain_adaptation(self, x, i, z_proto_list, is_feature=False):
        if self.domain_specific=='True':
            x = x.cuda()
            x = Variable(x)
        fast_parameters = list(self.feature.parameters())  # the first gradient calcuated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            if self.domain_specific == 'True':
                z_all, z_support, z_query = self.parse_feature(x, is_feature=False)
                scores,set_loss = self.domain_loss(z_proto_list, z_support, z_query, i)
            else:
                z_proto_list, z_support_list = self.outer_domain_proto(x)
                scores,set_loss = self.domain_loss(z_proto_list, z_support_list, i, i)
            grad = torch.autograd.grad(set_loss, fast_parameters,
                                       create_graph=True)  # build full graph support gradient of gradient
            if self.approx:
                grad = [g.detach() for g in
                        grad]  # do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            for k, weight in enumerate(self.feature.parameters()):
                # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k]  # create weight.fast
                else:
                    weight.fast = weight.fast - self.train_lr * grad[
                        k]  # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                fast_parameters.append(
                    weight.fast)  # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts

        if self.domain_specific=='True':
            scores = self.set_forward(x)
            y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
            if self.loss_type == 'mse':
                y_oh = utils.one_hot(y_query, self.n_way)
                y_oh = y_oh.cuda()
                loss = self.loss(scores, y_oh)
            else:
                y_query = y_query.cuda()
                loss = self.loss(scores, y_query)
            return scores,loss
        else:
            loss_all=[]
            for i in range(self.num_domain):
                x_i = x[i].cuda()
                x_i = Variable(x_i)
                scores = self.set_forward(x_i)
                y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
                if self.loss_type == 'mse':
                    y_oh = utils.one_hot(y_query, self.n_way)
                    y_oh = y_oh.cuda()
                    loss_all.append(self.loss(scores, y_oh))
                else:
                    y_query = y_query.cuda()
                    loss_all.append(self.loss(scores, y_query))
            return scores,sum(loss_all)/self.num_domain

# --- Convolution block used in the relation module ---
class RelationConvBlock(nn.Module):
    maml = False

    def __init__(self, indim, outdim, padding=0):
        super(RelationConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = backbone.Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = backbone.BatchNorm2d_fw(outdim, momentum=1, track_running_stats=False)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim, momentum=1, affine=True, track_running_stats=False)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]

        for layer in self.parametrized_layers:
            backbone.init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x.float())
        return out


# --- Relation module adopted in RelationNet ---
class RelationModule(nn.Module):
    maml = False

    def __init__(self, input_size, hidden_size, loss_type='mse'):
        super(RelationModule, self).__init__()

        self.loss_type = loss_type
        padding = 1 if (input_size[1] < 10) and (input_size[
                                                     2] < 10) else 0  # when using Resnet, conv map without avgpooling is 7x7, need padding in block to do pooling

        self.layer1 = RelationConvBlock(input_size[0] * 2, input_size[0], padding=padding)
        self.layer2 = RelationConvBlock(input_size[0], input_size[0], padding=padding)

        shrink_s = lambda s: int((int((s - 2 + 2 * padding) / 2) - 2 + 2 * padding) / 2)

        if self.maml:
            self.fc1 = backbone.Linear_fw(input_size[0] * shrink_s(input_size[1]) * shrink_s(input_size[2]),
                                          hidden_size)
            self.fc2 = backbone.Linear_fw(hidden_size, 1)
        else:
            self.fc1 = nn.Linear(input_size[0] * shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size)
            self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if self.loss_type == 'mse':
            out = torch.sigmoid(self.fc2(out))
        elif self.loss_type == 'softmax':
            out = self.fc2(out)

        return out