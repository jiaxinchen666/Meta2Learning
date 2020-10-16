from data.datamgr import SetDataManager
from options import parse_args, get_resume_file, load_warmup_state
from utils import log
import os

'''os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"'''

import torch
import h5py

from methods import backbone
from methods.backbone import model_dict
from data.datamgr import SimpleDataManager
from options import parse_args, get_best_file, get_assigned_file
from methods_ours.ours_protonet import Ours_protonet
from methods_ours.ours_gnnnet import Ours_gnn
from methods_ours.ours_relationnet import ours_RelationNet
from methods_ours.ours_mn import Ours_mn
import data.feature_loader as feat_loader
import random
import numpy as np


# extract and save image features
def save_features(model, data_loader, featurefile,params):
    f = h5py.File(featurefile, 'w')
    max_count = len(data_loader) * data_loader.batch_size
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    count = 0
    for i, (x, y) in enumerate(data_loader):
        if (i % 10) == 0:
            print('    {:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        feats=model(x)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
        all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count + feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count
    f.close()


# evaluate using features
def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=15):
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])
    z_all = torch.from_numpy(np.array(z_all))

    model.n_query = n_query
    scores = model.set_forward(z_all, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y) * 100
    return acc
def feature_evaluation_fungi(cl_data_file, model, n_way=5, n_support=5, n_query=15):
    for cl in list(cl_data_file):
        if len(cl_data_file[cl])<=n_support+n_query :
            cl_data_file.pop(cl)
    class_list=cl_data_file.keys()
    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])
    z_all = torch.from_numpy(np.array(z_all))

    model.n_query = n_query
    scores = model.set_forward(z_all, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y) * 100
    return acc
def train(base_datamgr, base_set, val_loader, model, start_epoch, stop_epoch, params, name):
    # for validation
    max_acc = 0
    trlog={}
    trlog['parser']=params
    trlog['val_acc']=[]
    trlog['test_acc']=[]
    log(params.checkpoint_dir + '/' + name + '_printlog.txt','parser:{}\n'.format(params))
    for write_dataset in base_set:
        log(params.checkpoint_dir + '/' + name + '_printlog.txt',write_dataset)

    # training
    if params.train_aug=='True':
        aug=True
    else:
        aug=False
    for epoch in range(start_epoch, stop_epoch):
        train_loader = [
            base_datamgr.get_data_loader(os.path.join(params.data_dir, dataset, 'base.json'), aug=aug) for
            dataset in base_set]
        # train loop
        model.train()
        model.train_loop(epoch, optimizer=model_optim, train_loaders=train_loader)
        # validate
        model.eval()
        with torch.no_grad():
            acc = model.test_loop(val_loader)
            trlog['val_acc'].append(acc)
        # save
        if acc > max_acc:
            print("best model! save...")
            max_acc = acc
            log(params.checkpoint_dir + '/' + name + '_printlog.txt', 'Best val:{:f}\n'.format(max_acc))
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save(model.state_dict(), outfile)
        else:
            print('GG!! best accuracy {:f}'.format(max_acc))
        if ((epoch + 1) % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch + 1))
            torch.save(model.state_dict(), outfile)
        outfile=os.path.join(params.checkpoint_dir,name+'_trlog')
        torch.save(trlog,outfile)
    return


# --- main function ---
if __name__ == '__main__':

    # set numpy random seed
    np.random.seed(10)

    # parse argument
    params = parse_args('train')
    if params.mode in ['train_and_test','onlytrain']:
        name = 'ours_shot_' + str(params.n_shot) + '_query_' + str(
            params.n_query) + '_' + params.method + '_' + params.testset +'_lr_'+str(params.lr)
        if params.domain_specific=='True':
            name=name
        elif params.domain_specific=='False':
            name = name+ '_nospecific'
        else:
            raise print('domain_specific is wrong')
        if params.train_aug=='True':
            name=name+'_aug'
        elif params.train_aug=='False':
            name=name
        else:
            raise print('train_aug is wrong')
        # output and tensorboard dir
        params.tf_dir = '%s/log/%s' % (params.save_dir, name)
        params.checkpoint_dir = '%s/checkmodels/%s' % (params.save_dir, name)
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        # dataloader
        print('\n--- prepare dataloader ---')
        print('  train with multiple seen domains (unseen domain: {})'.format(params.testset))
        datasets = params.dataset
        datasets.remove(params.testset)
        val_file = os.path.join(params.data_dir, 'miniImagenet', 'val.json')

        # model
        if 'Conv' in params.model:
            image_size = 84
        else:
            image_size = 224

        n_query = params.n_query
        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager(image_size,n_query=n_query, **train_few_shot_params)
        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size,n_query=n_query,**test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)

        backbone.FeatureWiseTransformation2d_fw.feature_augment = False
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.ResNet.maml = True
        if params.method=='protonet':
            model = Ours_protonet(model_dict[params.model], domain_specific=params.domain_specific,
                                  fine_tune='False', train_lr=params.lr,tf_path=params.tf_dir, **train_few_shot_params)

        elif params.method == 'matchingnet':
            model = Ours_mn(model_dict[params.model],domain_specific=params.domain_specific,
                                  fine_tune='False', train_lr=params.lr, tf_path=params.tf_dir, **train_few_shot_params)
        elif params.method in ['relationnet','relationnet_softmax']:
            if params.model == 'Conv4':
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6':
                feature_model = backbone.Conv6NP
            else:
                feature_model = model_dict[params.model]
            if params.method=='relationnet':
                loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
                model =ours_RelationNet(feature_model, domain_specific=params.domain_specific,train_lr=params.lr,
                                    loss_type=loss_type, fine_tune='False',tf_path=params.tf_dir,**train_few_shot_params)

        elif params.method == 'gnnnet':
            model = Ours_gnn(model_dict[params.model],domain_specific=params.domain_specific,
                             fine_tune='False',train_lr=params.lr,tf_path=params.tf_dir, **train_few_shot_params)
        else:
            raise ValueError('Unknown method')
        print('  train with {} framework'.format(params.method))
        print(model)

        model_optim = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.cuda()

        # resume training
        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch
        if params.resume != '':
            resume_file = get_resume_file('%s/checkmodels/%s' % (params.save_dir, params.resume), params.resume_epoch)
            print(resume_file)
            if resume_file is not None:
                start_epoch = model.resume(resume_file)
                print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
            else:
                raise ValueError('No resume file')
        # load pre-trained feature encoder
        else:
            model.feature.load_state_dict(
                load_warmup_state('%s/checkmodels/%s' % (params.save_dir, params.warmup), params.method), strict=False)

        # training
        print('\n--- start the training ---')
        train(base_datamgr, datasets, val_loader, model, start_epoch, stop_epoch, params,name)

    # parse argument
    params = parse_args('test')
    if params.mode in ['train_and_test','onlytest']:
        name = 'ours_shot_' + str(params.n_shot) + '_query_' + str(
            params.n_query) + '_' + params.method + '_' + params.testset+'_lr_'+str(params.lr)
        if params.domain_specific=='True':
            name=name
        elif params.domain_specific=='False':
            name = name+ '_nospecific'
        else:
            raise print('domain_specific is wrong')
        if params.train_aug=='True':
            name=name+'_aug'
        elif params.train_aug=='False':
            name=name
        else:
            raise print('train_aug is wrong')

        print('Testing! {} shots on {} dataset with {} epochs of {}({})'.format(params.n_shot, params.testset,
                                                                                params.save_epoch, name,
                                                                                params.method))
        remove_featurefile = True

        print('\nStage 1: saving features')
        # dataset
        print('  build dataset')
        if 'Conv' in params.model:
            image_size = 84
        else:
            image_size = 224
        split = params.split
        loadfile = os.path.join(params.data_dir, params.testset, split + '.json')
        datamgr = SimpleDataManager(image_size, batch_size=64)
        data_loader = datamgr.get_data_loader(loadfile, aug=False)

        print('  build feature encoder')
        # feature encoder

        checkpoint_dir = '%s/checkmodels/%s' % (params.save_dir, name)
        if params.save_epoch != -1:
            modelfile = get_assigned_file(checkpoint_dir, params.save_epoch)
        else:
            modelfile = get_best_file(checkpoint_dir)
        if params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4':
                model = backbone.Conv4NP()
            elif params.model == 'Conv6':
                model = backbone.Conv6NP()
            else:
                model = model_dict[params.model](flatten=False)
        else:
            model = model_dict[params.model]()
        model = model.cuda()
        '''tmp = torch.load(modelfile)
        try:
            state = tmp['state']
        except KeyError:
            state = tmp['model_state']
        except:
            raise
        state_keys = list(state.keys())
        for i, key in enumerate(state_keys):
            if "feature." in key and not 'gamma' in key and not 'beta' in key:
                newkey = key.replace("feature.", "")
                state[newkey] = state.pop(key)
            else:
                state.pop(key)'''
        state = torch.load(modelfile)
        state_keys = list(state.keys())
        for i, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.", "")
                state[newkey] = state.pop(key)
            else:
                state.pop(key)

        model.load_state_dict(state)
        model.eval()

        # save feature file
        print('  extract and save features...')
        if params.save_epoch != -1:
            featurefile = os.path.join(checkpoint_dir.replace("checkmodels", "features"),
                                       split + "_" + str(params.save_epoch) + ".hdf5")
        else:
            featurefile = os.path.join(checkpoint_dir.replace("checkmodels", "features"), split + ".hdf5")
        dirname = os.path.dirname(featurefile)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        save_features(model, data_loader, featurefile, params)

        print('\nStage 2: evaluate')
        acc_all = []
        iter_num = 1000
        few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        # model
        print('  build metric-based model')
        if params.method == 'protonet':
            model = Ours_protonet(model_dict[params.model],domain_specific=params.domain_specific,
                                  fine_tune='False',train_lr=params.lr, **few_shot_params)
        elif params.method == 'matchingnet':
            #backbone.LSTMCell.maml = False
            model = Ours_mn(model_dict[params.model],domain_specific=params.domain_specific,
                                  fine_tune='False', train_lr=params.lr,  **few_shot_params)
        elif params.method == 'gnnnet':
            model = Ours_gnn(model_dict[params.model],domain_specific=params.domain_specific,train_lr=params.lr,
                             fine_tune='False', **few_shot_params)

        elif params.method in ['relationnet']:
            if params.model == 'Conv4':
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6':
                feature_model = backbone.Conv6NP
            else:
                feature_model = model_dict[params.model]
            if params.method == 'relationnet':
                loss_type = 'mse'  # if params.method == 'relationnet' else 'softmax'
                model = ours_RelationNet(feature_model, domain_specific=params.domain_specific,train_lr=params.lr,
                                         loss_type=loss_type, fine_tune='False',
                                         **few_shot_params)
        else:
            raise ValueError('Unknown method')
        model = model.cuda()
        model.eval()

        # load model
        checkpoint_dir = '%s/checkmodels/%s' % (params.save_dir, name)
        if params.save_epoch != -1:
            modelfile = get_assigned_file(checkpoint_dir, params.save_epoch)
        else:
            modelfile = get_best_file(checkpoint_dir)
        if modelfile is not None:
            tmp = torch.load(modelfile)
            try:
                model.load_state_dict(tmp)
            except RuntimeError:
                print('warning! RuntimeError when load_state_dict()!')
                model.load_state_dict(tmp, strict=False)
            except KeyError:
                for k in tmp:  #### revise latter
                    if 'running' in k:
                        tmp[k] = tmp[k].squeeze()
                model.load_state_dict(tmp, strict=False)
            except:
                raise

        # load feature file
        print('  load saved feature file')
        cl_data_file = feat_loader.init_loader(featurefile)

        # start evaluate
        print('  evaluate')
        for i in range(iter_num):
            if params.testset=='fungi':
                acc = feature_evaluation_fungi(cl_data_file, model, n_query=15, **few_shot_params)
            else:
                acc = feature_evaluation(cl_data_file, model, n_query=15, **few_shot_params)
            acc_all.append(acc)

        # statics
        print('  get statics')
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('  %d test iterations: Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        # remove feature files [optional]
        if remove_featurefile:
            os.remove(featurefile)
        #log('./output/'+params.testset+'_acc.txt','Ours: %s %s Acc = %4.2f%% +- %4.2f%% shot:%d \n' % (params.method,params.domain_specific, acc_mean, 1.96 * acc_std /np.sqrt(iter_num),params.n_shot))
        log(checkpoint_dir + '/' + name + '_printlog.txt',
            '  %d test iterations: Acc = %4.2f%% +- %4.2f%%\n' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
