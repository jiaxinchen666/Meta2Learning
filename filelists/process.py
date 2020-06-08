import sys
import os
from subprocess import call

if len(sys.argv) != 2:
    raise Exception('Incorrect command! e.g., python3 process.py DATASET [cars, cub, places, miniImagenet, plantae]')
dataset = sys.argv[1]

print('--- process ' + dataset + ' dataset ---')
if not os.path.exists(os.path.join(dataset, 'source')):
    os.makedirs(os.path.join(dataset, 'source'))
os.chdir(os.path.join(dataset, 'source'))

# download files
if dataset == 'cars':
    call('wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz', shell=True)
    call('tar -zxf cars_train.tgz', shell=True)
    call('wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz', shell=True)
    call('tar -zxf car_devkit.tgz', shell=True)
elif dataset == 'cub':
    call('wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz', shell=True)
    call('tar -zxf CUB_200_2011.tgz', shell=True)
elif dataset == 'fungi':
    call('wget https://data.deic.dk/public.php?service=files&t=2fd47962a38e2a70570f3be027cea57f&download',shell=True)
    call('tar -zxf fungi_train_val.tgz', shell=True)
elif dataset == 'flowers':
    call('wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',shell=True)
    call('tar -zxf 102flowers.tgz')
    call('wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat',shell=True)
elif dataset == 'pets':
    call('wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz',shell=True)
    call('tar images.tar.gz')
elif dataset == 'miniImagenet':
  # this file is from MAML++: https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch
    call('wget http://vllab.ucmerced.edu/ym41608/projects/CrossDomainFewShot/filelists/mini_imagenet_full_size.tar.bz2', shell=True)
    call('tar -xjf mini_imagenet_full_size.tar.bz2', shell=True)
elif dataset == 'plantae':
    call('wget http://vllab.ucmerced.edu/ym41608/projects/CrossDomainFewShot/filelists/plantae.tar.gz', shell=True)
    call('tar -xzf plantae.tar.gz', shell=True)
else:
    raise Exception('No such dataset!')2d

# process file
os.chdir('..')
call('python3 write_' + dataset + '_filelist.py', shell=True)
