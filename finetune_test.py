import torch
import os
import h5py
from utils import log
from methods.backbone import model_dict
from data.datamgr import SetDataManager
from options import parse_args, get_best_file, get_assigned_file
from methods import backbone
from methods_ours.ours_protonet import Ours_protonet
from methods_ours.ours_relationnet import ours_RelationNet
from methods_ours.ours_gnnnet import Ours_gnn
from methods_ours.ours_mn import Ours_mn
import random
import numpy as np
# --- main ---
if __name__ == '__main__':
    # parse argument
    params = parse_args('test')
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
    print('Testing! {} shots on {} dataset with {} epochs of {}({})'.format(params.n_shot, params.testset,
                                                                            params.save_epoch, name,
                                                                            params.method))
    # dataset
    print('  build dataset')
    if 'Conv' in params.model:
        image_size = 84
    else:
        image_size = 224
    split = params.split
    loadfile = os.path.join(params.data_dir, params.testset, split + '.json')
    test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    val_datamgr = SetDataManager(image_size, n_query=params.n_query, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(loadfile, aug=False)

    datasets = params.dataset
    datasets.remove(params.testset)

    base_loaders = [
        val_datamgr.get_data_loader(os.path.join(params.data_dir, dataset, 'base.json'),
        aug=False) for dataset in datasets]


    print('  build feature encoder')
    # feature encoder
    checkpoint_dir = '%s/checkmodels/%s' % (params.save_dir, name)
    if params.save_epoch != -1:
        modelfile = get_assigned_file(checkpoint_dir, params.save_epoch)
    else:
        modelfile = get_best_file(checkpoint_dir)
    print('\nStage 2: evaluate')
    acc_all = []
    iter_num = 10
    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    # model
    backbone.FeatureWiseTransformation2d_fw.feature_augment = False
    backbone.ConvBlock.maml = True
    backbone.SimpleBlock.maml = True
    backbone.ResNet.maml = True
    if params.method=='protonet':
        model = Ours_protonet(model_dict[params.model],domain_specific=params.domain_specific,fine_tune='True',train_lr=params.lr,**few_shot_params)
    elif params.method=='gnnnet':
        model = Ours_gnn(model_dict[params.model],domain_specific=params.domain_specific,fine_tune='True',train_lr=params.lr, **few_shot_params)
    elif params.method == 'matchingnet':
        model = Ours_mn(model_dict[params.model],domain_specific=params.domain_specific,
                                  fine_tune='False', train_lr=params.lr,  **few_shot_params)
    elif params.method in ['relationnet','relationnet_softmax']:
        model=ours_RelationNet(model_dict[params.model],domain_specific=params.domain_specific,fine_tune='True',train_lr=params.lr, **few_shot_params)
    else:
        raise print('No such method')


    model = model.cuda()

    '''save_model = torch.load(modelfile)['model_state']
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)'''

    model.load_state_dict(torch.load(modelfile))
    
    for i in range(iter_num):
        acc = model.finetune_testloop(base_loaders=base_loaders,test_loader=val_loader)
        acc_all.append(acc)

    # statics
    print('  get statics')
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('Fine tune:  %d test iterations: Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
#     log('./output/'+params.testset+'_acc.txt','Finetune: %s %s Acc = %4.2f%% +- %4.2f%% shot: %d \n' % (params.method,params.domain_specific, acc_mean, 1.96 * acc_std /np.sqrt(iter_num),params.n_shot))

    log(checkpoint_dir + '/' + name + '_printlog.txt',
        'Fine tune:  %d test iterations: Acc = %4.2f%% +- %4.2f%%\n' % (
        iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))




