import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import openpyxl
import json
import random
import os
import pandas as pd
from skimage import io
from torchvision.transforms import RandomCrop, ToTensor, Grayscale, Pad, Compose, ToPILImage, CenterCrop, Resize
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import logging
import random

def prepare_online_data(args):
    data = pd.read_excel(args.data_path).to_dict()
    data_ = []
    for idx, index in data['ind'].items():
        image = io.imread(os.path.join('data', data['image_path'][idx]), as_gray=True)
        d = {
            'image': image,
            'uptake': data['uptake'][idx],
            'halflife': data['halflife'][idx],
            'dosage': data['dosage'][idx],
            'effect': data['effect'][idx]
        }
        data_.append(d)
    
    random.shuffle(data_)

    train_len = round(len(data_) * 0.8)
    train_, test_ = data_[:train_len], data_[train_len:]

    train_length = round(train_len * (1.0 / args.split_num))
    train_ = [train_[i * train_length: (i + 1) * train_length] for i in range(args.split_num)]

    train_lst = [[[data for data in train_[j] if data['effect'] == i] for i in [-2, -1, 0, 1, 2]] for j in range(args.split_num)]

    aug_num = [max([len(d) for d in train_lst[j]]) for j in range(args.split_num)]

    random.seed(args.seed)
    
    # print(aug_num)
    # print([[len(d) for d in train_lst[j]] for j in range(args.split_num)])

    for j in range(args.split_num):
        for i in [-2, -1, 0, 1, 2]:
            if len(train_lst[j][i]) < aug_num[j]:
                train_[j] += list(random.choices(train_lst[j][i], k=(aug_num[j] - len(train_lst[j][i]))))
    
    return train_, test_


def prepare_data(args):
    data = pd.read_excel(args.data_path).to_dict()
    data_ = []
    for idx, index in data['ind'].items():
        image = io.imread(os.path.join('data', data['image_path'][idx]), as_gray=True)
        d = {
            'image': image,
            'uptake': data['uptake'][idx],
            'halflife': data['halflife'][idx],
            'dosage': data['dosage'][idx],
            'effect': data['effect'][idx]
        }
        data_.append(d)
    
    random.shuffle(data_)

    train_len = round(len(data_) * 0.8)
    train_, test_ = data_[:train_len], data_[train_len:]

    train_lst = [[data for data in train_ if data['effect'] == i] for i in [-2, -1, 0, 1, 2]]

    aug_num = max([len(d) for d in train_lst])

    random.seed(args.seed)
    
    # print(aug_num)
    # print([len(d) for d in train_lst])

    for i in [-2, -1, 0, 1, 2]:
        if len(train_lst[i]) < aug_num:
            train_ += list(random.choices(train_lst[i], k=(aug_num - len(train_lst[i]))))
    
    return train_, test_

        
class myDataset(Dataset):
    def __init__(self, data):
        super(myDataset,self).__init__()
        self.transform = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-tiny-224")
        self.data = data
        
    def __getitem__(self,idx):
        data = self.data[idx]
        image = self.transform(data['image'][np.newaxis].repeat(3,0), return_tensor='pt')['pixel_values']
        feat = torch.tensor([
            data['uptake'], 
            data['halflife']]
        )
        label = torch.tensor([
            data['dosage'], 
            data['effect']
        ])
        return image, feat, label
    
    def __len__(self):
        return len(self.data)
    
def statistics(dataset):
    label = {key: 0 for key in [-2, -1, 0, 1, 2]}

    for d in dataset.data:
        label[d['effect']] += 1
    return label
        
def get_dataset(args):
    train, test = prepare_data(args)
    train, test = myDataset(train), myDataset(test)

    print("train:", statistics(train))
    print("test:", statistics(test))

    return train, test

def get_online_dataset(args):
    train, test = prepare_online_data(args)
    train, test = [myDataset(train[j]) for j in range(args.split_num)], myDataset(test)

    print("train:", [statistics(train[j]) for j in range(args.split_num)])
    print("test:", statistics(test))

    #train, test = random_split(dataset, [train_len, size-train_len])

    return train, test
