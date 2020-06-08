import torch.nn as nn
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch
import torch.nn.functional as F

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support,domain_specific,fine_tune,train_lr,flatten=True,tf_path=None, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.feature = model_func(flatten=flatten)  # , leakyrelu=leakyrelu)
        self.feat_dim = self.feature.final_feat_dim
        self.change_way = change_way  # some methods allow different_way classification during training and test
        self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None
        self.loss_fn = nn.CrossEntropyLoss()
        self.approx = True
        self.train_lr = train_lr
        self.domain_specific=domain_specific
        self.fine_tune=fine_tune
        if self.fine_tune=='True':
            self.num_domain=4
        else:
            self.num_domain=3
        if self.fine_tune=='True':
            self.task_update_num = 5
        else:
            self.task_update_num=5

    @abstractmethod
    def set_forward(self, x):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass
    
    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        x = x.cuda()
        if is_feature:
            z_all = x
        else:
            # print(self.n_support,self.n_query)
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all=self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        return z_all, z_support, z_query

    def domain_loss(self, z_proto_list, z_support, z_all, i):
        if self.domain_specific=='True':
            z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
            z_proto = z_all.view(self.n_way, self.n_query, -1).mean(1).mean(0)
            z_proto_list[i] = z_proto
            domain_proto = torch.stack(z_proto_list)
            dists = euclidean_dist(z_support, domain_proto)
            scores = -dists
            y_support = torch.from_numpy(np.repeat(i, self.n_support * self.n_way))
            y_support = y_support.cuda()
            loss = self.loss_fn(scores, y_support)
        else:
            domain_proto = torch.stack(z_proto_list)
            loss_all=[]
            for i in range(self.num_domain):
                dists = euclidean_dist(z_support[i].cuda(), domain_proto)
                scores = -dists
                y_support = torch.from_numpy(np.repeat(i, self.n_support * self.n_way))
                y_support = y_support.cuda()
                loss_all.append(self.loss_fn(scores, y_support))
            loss=sum(loss_all)/self.num_domain
        return scores,loss

    def domain_adaptation(self, x, i, z_proto_list, is_feature=False):
        if self.domain_specific == 'True':
            x = x.cuda()
            x = Variable(x)
        fast_parameters = list(self.feature.parameters())  # the first gradient calcuated in line 45 is based on original weight
        for weight in self.feature.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            if self.domain_specific=='True':
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
        if self.fine_tune=='True':
            self.eval()
        if self.domain_specific=='True':
            scores = self.set_forward(x)
            y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
            y_query = y_query.cuda()
            loss = self.loss_fn(scores, y_query)
            return scores,loss
        else:
            loss_all=[]
            for i in range(self.num_domain):
                x_i = x[i].cuda()
                x_i = Variable(x_i)
                scores = self.set_forward(x_i)
                y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
                y_query = y_query.cuda()
                loss_all.append(self.loss_fn(scores, y_query))
            loss=sum(loss_all)/self.num_domain
            return scores,loss


    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float( top1_correct) / len(y_query)


    def train_loop(self, epoch, train_loaders, optimizer):
        avg_loss = 0
        optimizer.zero_grad()
        loss_all = []
        train_loader = zip(train_loaders[0], train_loaders[1], train_loaders[2])
        for i, data in enumerate(train_loader):
            self.n_query = list(data)[0][1].size(1) - self.n_support
            self.num_sample=self.num_domain * self.n_way * (self.n_query + self.n_support)
            data_list = [x for (x, _) in list(data)]
            if self.domain_specific=='True':
                z_proto_list=self.outer_domain_proto(data_list)
                for j in range(self.num_domain):
                    _, loss = self.domain_adaptation(data_list[j], j, z_proto_list)
                    loss_all.append(loss)
                loss = torch.stack(loss_all).sum(0) / self.num_domain
                loss_all=[]
            else:
                j=0
                _, loss = self.domain_adaptation(data_list, j, j)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            optimizer.zero_grad()
        print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / (i + 1)))
        return avg_loss

    def test_loop(self, test_loader, record=None):
        self.eval()
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            correct_this = self.correct(x)
            acc_all.append(correct_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        return acc_mean

    def finetune_testloop(self,base_loaders, test_loader):
        train_loader = zip(base_loaders[0], base_loaders[1], base_loaders[2],test_loader)
        acc_all=[]
        for i, data in enumerate(train_loader):
            self.n_query = list(data)[0][1].size(1) - self.n_support
            data_list = [x for (x, _) in list(data)]
            self.domain_specific='True'
            z_proto_list = self.outer_domain_proto(data_list)
            scores,_ = self.domain_adaptation(data_list[self.num_domain-1], self.num_domain-1, z_proto_list)
            y_query = np.repeat(range(self.n_way), self.n_query)
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:, 0] == y_query)
            acc_all.append(float(top1_correct) / len(y_query)*100)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        #print('---Finetune %d Test Acc = %4.2f%% +- %4.2f%% ---' % (i+1, acc_mean, 1.96 * acc_std / np.sqrt(i+1)))
        return acc_mean

    def outer_domain_proto(self, data_list, is_feature=False):
        z_proto_list = []
        z_support_list = []
        for i in range(len(data_list)):
            z_all, z_support, z_query = self.parse_feature(data_list[i], is_feature)
            z_support=z_support.contiguous().view(self.n_way*self.n_support,-1)
            z_proto = z_query.view(self.n_way, self.n_query, -1).mean(1).mean(
                0)  # the shape of z is [n_data, n_dim]
            z_proto_list.append(z_proto)
            z_support_list.append(z_support)
        if self.domain_specific == 'True':
            z_proto_list = [z_proto.detach() for z_proto in z_proto_list]
            return z_proto_list
        elif self.domain_specific == 'False':
            return z_proto_list,z_support_list

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
