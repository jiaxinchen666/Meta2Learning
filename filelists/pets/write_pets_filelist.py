import numpy as np
#from os import listdir
from os.path import join#isfile, isdir, join
import os
#import json
import random
from scipy.io import loadmat

cwd = os.getcwd()
data_path = join(cwd,'./images')
savedir = './'
dataset_list = ['base','val','novel']

#data_list = np.array(loadmat('./imagelabels.mat')['labels'])


labels=[]
files=os.listdir(data_path)
for file in files:
    if '.jpg' in file:
        label,image=file.rsplit('_',1)
        if label not in labels:
            labels.append(label)
        else:
            pass
print(labels)
classfile_list_all = [[] for i in range(len(labels))]
for file in files:
    if '.jpg' in file:
        label,image=file.rsplit('_',1)
        image='/data/glusterfs_sharing_04_v3/11117638/CrossDomainFewShot/filelists/pets/images/'+file
        classfile_list_all[labels.index(label)].append(image)
    else:
        pass

for i in range(len(classfile_list_all)):
    random.shuffle(classfile_list_all[i])

for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'base' in dataset:
            if (i%2 == 0):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset:
            if (i%4 == 1):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'novel' in dataset:
            if (i%4 == 3):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%d",' % item  for item in range(len(labels))])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
