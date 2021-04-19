A pytorch implementation of Domain-Oriented Meta-Learning for Cross-Domain Few-shot Classification.

### Prerequisites
- Python >= 3.5
- Pytorch >= 1.3 and torchvision (https://pytorch.org/)
- You can use the `requirements.txt` file we provide to setup the environment via Anaconda.
```
conda create --name py36 python=3.6
conda install pytorch torchvision -c pytorch
pip3 install -r requirements.txt
```

### Install
Clone this repository:
```
git clone https://github.com/jiaxinchen666/DOM.git
cd CrossDomainFewShot
```

### Datasets

- Set `DATASET_NAME` to: `cars`, `cub`, `miniImagenet`, `flowers`, `pets`, `fungi`, or `plantae`.

```
cd filelists
python process.py DATASET_NAME
cd ..
```

For datasets 'vegetable', 'food', and 'butterfly', download [vegetable](https://github.com/ustc-vim/vegfru), [butterfly](https://www.dropbox.com/sh/3p4x1oc5efknd69/AABwnyoH2EKi6H9Emcyd0pXCa?dl=0),
[food](https://www.kaggle.com/dansbecker/food-101) and extract them into ./filelists/DATASET_NAME

```
cd filelists
python write_DATASET_filelist.py
cd ..
```

### Feature encoder pre-training
We adopt baseline++ for MatchingNet, and baseline from CloserLookFewShot for other metric-based frameworks.

Download the pre-trained feature encoders.
```
cd output/checkpoints
python download_encoder.py
cd ../..
```

### Training

--dataset A DATASET LIST CONSISTING OF TRAINING DOMAINS AND TEST DOMAIN 

--n_shot 1/5, --method 'protonet'/'relationnet'/'gnnnet'/'matchingnet', --domain_specific 'True'/'False'(only for DOM)

Baseline

```
python train_baseline.py --n_shot 5 --testset DATASET_NAME --method METHOD --n_query 16 --mode 'onlytrain';
```

LFT (Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation)

```
python train.py --n_shot 5 --testset DATASET_NAME --method METHOD --n_query 16 --mode 'onlytrain';
```

DOM

```
python train_ours.py --n_shot 5 --testset DATASET_NAME --method METHOD --n_query 16 --mode 'onlytrain' --domain_specific 'True' --lr '0.1';
```

### Evaluation

Baseline

```
python train_baseline.py --n_shot 5 --testset DATASET_NAME --method METHOD --n_query 16 --mode 'onlytest';
```

LFT (Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation)

```
python train.py --n_shot 5 --testset DATASET_NAME --method METHOD --n_query 16 --mode 'onlytest';
```

DOM

```
python train_ours.py --n_shot 5 --testset DATASET_NAME --method METHOD --n_query 16 --mode 'onlytest' --domain_specific 'True' --lr '0.1';
```

DOM (Further adaptation)

```
python finetune_test.py --n_shot 5 --testset DATASET_NAME --method METHOD --n_query 16; 
```

## Note
- This code is built upon the implementation from [CrossDomainFewShot](https://github.com/hytseng0509/CrossDomainFewShot) and [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot).
- The dataset, model, and code are for non-commercial research purposes only.
