from sklearn.ensemble import RandomForestRegressor
import torch
import os
import random
import numpy as np
import pandas as pd
from ztools import from_smiles

def process(x, y):
    x_temp = x.drop(['membrane_smiles', 'solvent_smiles', 'solute_smiles', 'full_smiles'], axis=1)
    data_list = []
    d1 = []
    d2 = []
    columns_list = x_temp.columns
    for index in range(len(x)):
        permeance = y.iloc[index]['permeance']
        rejection = y.iloc[index]['rejection']

        y_temp = torch.tensor([permeance, rejection], dtype=torch.float).view(1, -1)
        data_sum = from_smiles(x.iloc[index]['solute_smiles'])
        # data_temp = from_smiles(X.iloc[index]['solute_smiles'])
        data_sum.y = y_temp
        # data_sum.x_solute = data_temp.x
        # data_sum.edge_index_solute = data_temp.edge_index
        # data_sum.edge_attr_solute = data_temp.edge_attr
        # data_sum.smiles_solute = data_temp.smiles
        data_sum.tb = torch.tensor(x_temp.iloc[index][columns_list], dtype=torch.float).view(1, -1)
        data_list.append(data_sum)
        # dt1=from_smiles(X.iloc[index]['Solvent Smile'])
        # dt2=from_smiles(X.iloc[index]['Solute smile'])
        # dt1.y=y
        # dt2.y=y
        # d1.append(dt1)
        # d2.append(dt2)
    return data_list
def mix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def drop_smiles_split_rf_evaluate(data_new, target, method, seed, x_train, x_test, y_train, y_test):
    # data = data_new.copy()
    # data.drop(['Solvent Smile', 'Solute smile'], axis=1, inplace=True)
    # data = pd.get_dummies(data)
    # data = data.drop_duplicates()
    # y_columns = target
    # y = data[y_columns]
    # x = data.drop(y_columns, axis=1)
    # if method == 'z':
    #     x = (x - x.mean()) / (x.std())
    # elif method == 'max_min':
    #     x = (x - x.min()) / (x.max() - x.min())
    # else:
    #     x = x
    y_temp = data_new[target]
    x_temp = data_new.drop(target, axis=1)
    x_train = pd.concat([x_train, x_temp], axis=0)
    y_train = pd.concat([y_train, y_temp], axis=0)
    x_train = make_index(x_train)
    y_train = make_index(y_train)

    # x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.10, random_state=seed)
    #
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, shuffle=True, test_size=0.111,
    #                                                   random_state=seed)
    # 5394 ，674， 675
    # x_train = x_train.drop(['Solvent Smile', 'Solute smile'], axis=1).copy()
    # x_test = x_test.drop(['Solvent Smile', 'Solute smile'], axis=1).copy()
    # x_val = x_val.drop(['Solvent Smile', 'Solute smile'], axis=1).copy()
    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)
    r_train = rf.score(x_train, y_train)
    r_test = rf.score(x_test, y_test)
    # r_val = rf.score(x_val, y_val)
    print(f"train:{r_train},test:{r_test}")
    return r_test


def make_index(data):
    index = []
    for i in range(data.shape[0]):
        index.append(i)
    data.index = index
    return data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
