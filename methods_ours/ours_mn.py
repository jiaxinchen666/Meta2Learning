# This code is modified from https://github.com/jakesnell/prototypical-networks

import torch
import torch.nn as nn
import numpy as np
from methods_ours.meta_template import MetaTemplate
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from methods import backbone
import utils
class Ours_mn(MetaTemplate):
    def __init__(self,model_func, n_way, n_support,domain_specific,fine_tune,train_lr,tf_path=None):
        super(Ours_mn, self).__init__(model_func, n_way, n_support,                                 domain_specific=domain_specific,fine_tune=fine_tune,train_lr=train_lr, tf_path=None)
        self.proba=proba

    # loss function
        self.loss_fn    = nn.NLLLoss()

        # metric
        self.FCE = FullyContextualEmbedding(self.feat_dim)
        self.G_encoder = backbone.LSTM(self.feat_dim, self.feat_dim, 1, batch_first=True, bidirectional=True)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.method = 'MatchingNet'

    def encode_training_set(self, S, G_encoder = None):
        if G_encoder is None:
            G_encoder = self.G_encoder
        out_G = G_encoder(S.unsqueeze(0))
        out_G = out_G.squeeze(0)
        G = S + out_G[:,:S.size(1)] + out_G[:,S.size(1):]
        G_norm = torch.norm(G,p=2, dim =1).unsqueeze(1).expand_as(G)
        G_normalized = G.div(G_norm+ 0.00001)
        return G, G_normalized

    def get_logprobs(self, f, G, G_normalized, Y_S, FCE = None):
        if FCE is None:
            FCE = self.FCE
        F = FCE(f, G)
        F_norm = torch.norm(F,p=2, dim =1).unsqueeze(1).expand_as(F)
        F_normalized = F.div(F_norm + 0.00001)
        scores = self.relu( F_normalized.mm(G_normalized.transpose(0,1))  ) *100 # The original paper use cosine simlarity, but here we scale it by 100 to strengthen highest probability after softmax
        softmax = self.softmax(scores)
        logprobs =(softmax.mm(Y_S)+1e-6).log()
        return logprobs

    def set_forward(self, x, is_feature = False):
        _,z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view( self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view( self.n_way* self.n_query, -1 )
        G, G_normalized = self.encode_training_set( z_support)

        y_s         = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        Y_S         = utils.one_hot(y_s, self.n_way).cuda()
        f           = z_query
        logprobs = self.get_logprobs(f, G, G_normalized, Y_S)
        return logprobs

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = y_query.cuda()

        logprobs = self.set_forward(x)
        loss = self.loss_fn(logprobs, y_query)

        return logprobs, loss

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
            y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
            logprobs = self.set_forward(x)
            loss = self.loss_fn(logprobs, y_query)
            return scores,loss
        else:
            loss_all=[]
            for i in range(self.num_domain):
                x_i = x[i].cuda()
                x_i = Variable(x_i)
                scores = self.set_forward(x_i)
                y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
                y_query = y_query.cuda()
                logprobs = self.set_forward(x_i)
                loss_all.append(self.loss_fn(logprobs, y_query))
            loss=sum(loss_all)/self.num_domain
            return scores,sum(loss_all)/self.num_domain

            

    def cuda(self):
        super(Ours_mn, self).cuda()
        self.FCE = self.FCE.cuda()
        return self

# --- Fully contextual embedding function adopted in Matchingnet ---
class FullyContextualEmbedding(nn.Module):
    def __init__(self, feat_dim):
        super(FullyContextualEmbedding, self).__init__()
        self.lstmcell = backbone.LSTMCell(feat_dim*2, feat_dim)
        self.softmax = nn.Softmax(dim=1)
        self.c_0 = torch.zeros(1, feat_dim)
        self.feat_dim = feat_dim

    def forward(self, f, G):
        h = f
        c = self.c_0.expand_as(f)
        G_T = G.transpose(0,1)
        K = G.size(0) #Tuna to be comfirmed
        for k in range(K):
            logit_a = h.mm(G_T)
            a = self.softmax(logit_a)
            r = a.mm(G)
            x = torch.cat((f, r),1)

            h, c = self.lstmcell(x, (h, c))
            h = h + f
        return h

    def cuda(self):
        super(FullyContextualEmbedding, self).cuda()
        self.c_0 = self.c_0.cuda()
        return self



