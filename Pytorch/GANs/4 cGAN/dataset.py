import pandas as pd
from torch.utils.data import (
    Dataset,
    DataLoader,
)
import torch.nn as nn
import torch
from torchaudio import transforms


class APGDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x_data = self.data.iloc[item, 0:21]
        x_data = torch.tensor(x_data)
        y_label = torch.tensor(int(self.data.iloc[item, 21]))

        return (x_data.float(), y_label)

def write_to_total():
    # Import training data
    dataset_train = pd.read_csv("dataset/APG/csvDataFeaturesTrain.csv", delimiter=';')
    dataset_test = pd.read_csv("dataset/APG/csvDataFeaturesTest.csv", delimiter=';')
    X = dataset_train.append(dataset_test)
    X.groupby('attack').count()
    X['attack'] = X['attack'].map({0:0, 1:1, 2:1, 3:1, 4:1})

    X.to_csv("dataset/APG/total.csv", index=False)

def test():
    dataset = APGDataset(csv_file="dataset/APG/total.csv", root_dir="dataset/APG/", transforms=transforms.Tensor())
    dataloader = DataLoader(dataset=dataset, batch_size=4096, shuffle=True)

    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device="cpu")
        labels = targets.to(device="cpu")

        print(data.shape, labels.shape)

#test()