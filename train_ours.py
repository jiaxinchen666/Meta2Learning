from tsnecuda import TSNE
import feature_loader as feat_loader
import numpy
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
method='matchingnet'
datasets=['cub','cars','fungi']
base_features=[[] for dataset in datasets]
y_base=[[] for dataset in datasets]
for index in range(len(datasets)):
    dataset=datasets[index]
    #featurefile='./features/original_shot_5_query_16_'+method+'_flowers_aug/'+dataset+'_lft_'+method+'_base.hdf5'
    featurefile='./features/ours_shot_5_query_16_'+method+'_flowers_lr_0.1_aug/'+dataset+'_ours_'+method+'_base.hdf5'
    #featurefile='./features/baseline_shot_5_query_16_'+method+'_cars_aug/'+dataset+'_baseline_'+method+'_base.hdf5'
    cl_data_file = feat_loader.init_loader(featurefile)
    for key in cl_data_file.keys():
        base_features[index].extend(cl_data_file[key])
        y_base[index].extend([index for y in range(len(cl_data_file[key]))])
novel_features=[]
novel_y=[]
#featurefile='./features/original_shot_5_query_16_'+method+'_flowers_aug/flowers_lft_'+method+'_novel.hdf5'
featurefile='./features/ours_shot_5_query_16_'+method+'_flowers_lr_0.1_aug/flowers_ours_'+method+'_novel.hdf5'
#featurefile='./features/baseline_shot_5_query_16_protonet_cars_aug/cars_baseline_protonet_novel.hdf5'
cl_data_file = feat_loader.init_loader(featurefile)
for key in cl_data_file.keys():
    novel_features.extend(cl_data_file[key])
    novel_y.extend([3 for y in range(len(cl_data_file[key]))])

feature_array=[0 for i in range(3)]

for index in range(3):
    features=base_features[index][0:2000]
    for i in range(2000):
        feature=features[i]
        if i==0:
            feature_array[index]=np.expand_dims(numpy.array(feature),axis=0)
        else:
            feature_array[index]=np.concatenate((feature_array[index],np.expand_dims(numpy.array(feature),axis=0)),axis=0)

X=np.concatenate((feature_array),axis=0)
novel_features=novel_features[0:2000]
for i in range(2000):
    feature = novel_features[i]
    if i == 0:
        novel_feature_array= np.expand_dims(numpy.array(feature), axis=0)
    else:
        novel_feature_array = np.concatenate((novel_feature_array, np.expand_dims(numpy.array(feature), axis=0)),
                                              axis=0)
X=np.concatenate((X,novel_feature_array),axis=0)

X_embedded =manifold.TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)

#X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)
X=list(X_embedded[0:2000].T[0])
Y=list(X_embedded[0:2000].T[1])
plt.plot(X,Y,'.',color='blue')
X=list(X_embedded[2000:4000].T[0])
Y=list(X_embedded[2000:4000].T[1])
plt.plot(X,Y,'.',color='red')
X=list(X_embedded[4000:6000].T[0])
Y=list(X_embedded[4000:6000].T[1])
plt.plot(X,Y,'.',color='yellow')
X=list(X_embedded[6000:8000].T[0])
Y=list(X_embedded[6000:8000].T[1])
plt.plot(X,Y,'.',color='green')
plt.show()