from torch.utils.data import Dataset, DataLoader

import torch.utils.data as data_utils

import os

import csv
import torch
import pickle

import random

from sklearn.metrics import roc_auc_score


class amDataloader(data_utils.Dataset):

    def __init__(self, x_train, y_train, cate, cont):
        self.data_list = []

        self.metric_name = 'AUC'
        self.cate = x_train[cate]
        self.cont = x_train[cont]
        self.label = y_train

    def __getitem__(self, idx):
        return (torch.tensor(self.cate.iloc[idx].values, dtype=torch.long),
                torch.tensor(self.cont.iloc[idx].values, dtype=torch.float32),
                torch.tensor(self.label.iloc[idx], dtype=torch.float32))

    def evaluate(self, label, pred):
        return roc_auc_score(label.item(), pred[:, 1].item())

    def __len__(self):
        return len(self.cont)


class MyDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.labels = x_train
        self.features = y_train

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features.iloc[idx].values, dtype=torch.float32), torch.tensor(self.labels.iloc[idx],
                                                                                               dtype=torch.long)


class make_split():

    def __init__(self):
        self.data_list = []

        file = '/home/ceyu.cy2/datasets/tabular/Income/income_evaluation.csv'
        now = 0
        with open(file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                self.data_list.append(row)

        cols = [[] for _ in range(15)]
        for each in self.data_list:
            for i in range(15):
                cols[i].append(each[i])

        cont = []
        cate = []
        mapping = [{} for _ in range(15)]
        for i in range(15):
            if i == 14:

                for j, each in enumerate(set(cols[i])):
                    mapping[i][each] = j
            elif i in [0, 2, 4, 10, 11, 12]:
                cont.append(cols[i])
            else:
                lens = len(set(cols[i]))

                for j, each in enumerate(set(cols[i])):
                    if each in mapping[i]:
                        raise ValueError
                    else:
                        mapping[i][each] = now + j
                now += lens

        n = 1
        for i in range(len(self.data_list)):
            j = 14
            if mapping[j] != {}:
                self.data_list[i][j] = mapping[j][self.data_list[i][j]]

        self.label = [each[14] for each in self.data_list]

        negative = [i for i in range(len(self.label)) if self.label[i] == 0]
        positive = [i for i in range(len(self.label)) if self.label[i] == 1]

        random.shuffle(negative)
        random.shuffle(positive)

        train = positive[:int(0.65 * len(positive))]
        train += negative[:int(0.65 * len(negative))]

        val = positive[int(0.65 * len(positive)): int(0.8 * len(positive))]
        val += negative[int(0.65 * len(negative)): int(0.8 * len(negative))]

        test = positive[int(0.8 * len(positive)):]
        test += negative[int(0.8 * len(negative)):]

        root = '/home/ceyu.cy2/datasets/tabular/Income/train_val_test/split4/'
        if not os.path.exists(root):
            os.mkdir(root)
        with open(root + 'train.pkl', 'wb') as f:
            pickle.dump(train, f)

        with open(root + 'val.pkl', 'wb') as f:
            pickle.dump(val, f)

        with open(root + 'test.pkl', 'wb') as f:
            pickle.dump(test, f)

# # 实例化Dataset
# dataset = MyDataset('data.csv')
#
# # 创建DataLoader
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
