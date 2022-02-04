import pandas as pd
from torch.utils.data import (
    Dataset,
    DataLoader,
)
import torch.nn as nn
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class APGDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.orig = pd.read_csv(csv_file)
        print(self.orig.shape)
        self.root_dir = root_dir
        #self.transforms = transforms.Compose([
        #        transforms.ToTensor(),
        #])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x_data = self.data[item, 0:21]
        x_data = torch.tensor(x_data)
        y_label = torch.tensor(int(self.data[item, 21]))

        return x_data.float(), y_label

def to_csv_attack_and_normal():
    dataset_train = pd.read_csv("dataset/APG/csvDataFeaturesTrain.csv", delimiter=';')
    dataset_test = pd.read_csv("dataset/APG/csvDataFeaturesTest.csv", delimiter=';')
    x = dataset_train.append(dataset_test)
    x.groupby('attack').count()
    x['attack'] = x['attack'].map({0:0, 1:1, 2:1, 3:1, 4:1})

    x_normal = x[x['attack'] == 0]
    x_attack = x[x['attack'] > 0]

    #x_normal = x_normal.drop(['attack'], axis=1)
    #x_attack = x_attack.drop(['attack'], axis=1)
    pd.DataFrame(x_normal).to_csv("dataset/APG/normal.csv", index=False)
    pd.DataFrame(x_attack).to_csv("dataset/APG/attacks.csv", index=False)

def test():
    dataset = APGDataset(csv_file="dataset/APG/normal.csv", root_dir="dataset/APG/")
    loader = DataLoader(dataset)

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device="cpu")
        print(data)
    #    labels = targets.to(device="cpu")

#to_csv_attack_and_normal()
test()