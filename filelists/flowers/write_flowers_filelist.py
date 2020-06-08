import numpy as np
#from os import listdir
from os.path import join#isfile, isdir, join
import os
#import json
import random
from scipy.io import loadmat

cwd = os.getcwd()
data_path = join(cwd,'./jpg')
savedir = './'
dataset_list = ['base','val','novel']

data_list = np.array(loadmat('./imagelabels.mat')['labels'])

classfile_list_all = [[] for i in range(len(list(set(data_list[0]))))]

for num,label in enumerate(list(data_list[0])):
    image_num=num+1
    if image_num <10:
        image='/data/glusterfs_sharing_04_v3/11117638/CrossDomainFewShot/filelists/flowers/jpg/image_0000'+str(image_num)+'.jpg'
    elif 10<=image_num<100:
        image = '/data/glusterfs_sharing_04_v3/11117638/CrossDomainFewShot/filelists/flowers/jpg/image_000' + str(
            image_num) + '.jpg'
    elif 100<=image_num<1000:
        image = '/data/glusterfs_sharing_04_v3/11117638/CrossDomainFewShot/filelists/flowers/jpg/image_00' + str(
            image_num) + '.jpg'
    else:
        image = '/data/glusterfs_sharing_04_v3/11117638/CrossDomainFewShot/filelists/flowers/jpg/image_0' + str(
            image_num) + '.jpg'
    classfile_list_all[label-1].append(image)


for i in range(len(classfile_list_all)):
    random.shuffle(classfile_list_all[i])
'''folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])'''

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
    fo.writelines(['"%d",' % item  for item in list(set(data_list[0]))])
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
