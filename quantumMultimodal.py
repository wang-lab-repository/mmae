import sys
import os
from get_data import generate_multimodal_data
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from ztools import from_smiles
import random
import numpy as np
import datetime
from utils import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

def mix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def process(x, y):
    x_temp = x.drop(['membrane_smiles', 'solvent_smiles', 'solute_smiles', 'full_smiles'], axis=1)
    data_list = []
    columns_list = x_temp.columns
    for index in range(len(x)):
        permeance = y.iloc[index]['permeance']
        rejection = y.iloc[index]['rejection']
        y_temp = torch.tensor([permeance, rejection], dtype=torch.float).view(1, -1)
        data_sum = from_smiles(x.iloc[index]['solute_smiles'])
        data_sum.y = y_temp
        data_sum.tb = torch.tensor(x_temp.iloc[index][columns_list], dtype=torch.float).view(1, -1)
        data_list.append(data_sum)
    return data_list


def get_p_value(path="bysj.xlsx", target=['permeance', 'rejection'], seed=123456, method='z',
                is_df=True,
                factor=1.0, delete_smile=False, pca=True):
    # x_train, x_val, x_test, y_train, y_val, y_test
    x_train, x_test, y_train, y_test = generate_multimodal_data(path=path,
                                                                method=method,
                                                                is_df=is_df,
                                                                factor=factor,
                                                                target=target,
                                                                delete_smile=delete_smile,
                                                                pca=pca,
                                                                seed=seed, is_rfe=True)
    return x_train, x_test, y_train, y_test


from q_model import q_gnnmodel, QuCrossModel


class model(nn.Module):
    def __init__(self, dropout_rate=0.10, f1=70, nlayers=1):
        super(model, self).__init__()
        dropout_rate = dropout_rate
        f1 = f1
        nlayers = nlayers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnnmodel = q_gnnmodel(f1=f1, dropout_rate=dropout_rate)
        self.tabularmodel = QuCrossModel(f1=f1, dropout_rate=dropout_rate, nlayers=nlayers)
        self.f1 = nn.Linear(8, f1)
        self.f2 = nn.Linear(f1, f1)
        self.f3 = nn.Linear(f1, 2)
        self.drop = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.weight = nn.Parameter(torch.randn(1))  # 定义一个可学习的权重参数

    def forward(self, data):  # 输入：9个特征
        data = data.to(self.device)
        gnn_x = self.gnnmodel(data)
        tabular_x = self.tabularmodel(data.tb)
        x = self.weight * gnn_x + (1 - self.weight) * tabular_x
        x = self.f1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.f2(x)
        x = self.relu(x)
        x = self.f3(x)
        return x


def train(seed=123456, epoch=1000, lr=0.001, dropout_rate=0.01, f1=100, nlayers=1, batch_size=128,
          path="bysj.xlsx", is_df=True, delete_smile=False, pca=True):
    mix_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x_train, x_test, y_train, y_test = get_p_value(path=path, seed=seed,
                                                   is_df=is_df, delete_smile=delete_smile, pca=pca)
    train_list = process(x=x_train, y=y_train)
    # val_list = process(x=x_val, y=y_val)
    test_list = process(x=x_test, y=y_test)
    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    train_loader1 = DataLoader(train_list, batch_size=len(train_list), shuffle=True)
    # val_loader = DataLoader(val_list, batch_size=len(val_list), shuffle=False)
    test_loader = DataLoader(test_list, batch_size=len(test_list), shuffle=False)
    epoch = epoch
    lr = lr
    net = model(dropout_rate=dropout_rate, f1=f1, nlayers=nlayers)  # 定义模型
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()
    early_stopping = EarlyStopping(patience=50, delta=0)
    for i in range(epoch):
        for data in train_loader:
            train_pre = net(data)
            train_loss = loss_func(train_pre, data.y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        # for k in val_loader:
        #     prediction_val = net(k)
        #     r2_val_1 = r2_score(k.y.cpu().detach().numpy()[:, 0], prediction_val.cpu().detach().numpy()[:, 0])
        #     r2_val_2 = r2_score(k.y.cpu().detach().numpy()[:, 1], prediction_val.cpu().detach().numpy()[:, 1])
        #     early_stopping(-(r2_val_1+r2_val_2))
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break
    net.eval()
    print("------------------------结果------------------------")
    # 计算训练集上的R²
    for i in train_loader1:
        prediction_train = net(i)
        r2_train_1 = r2_score(i.y.cpu().detach().numpy()[:, 0], prediction_train.cpu().detach().numpy()[:, 0])
        r2_train_2 = r2_score(i.y.cpu().detach().numpy()[:, 1], prediction_train.cpu().detach().numpy()[:, 1])
        rmse_train1 = np.sqrt(
            mean_squared_error(i.y.cpu().detach().numpy()[:, 0], prediction_train.cpu().detach().numpy()[:, 0]))
        rmse_train2 = np.sqrt(
            mean_squared_error(i.y.cpu().detach().numpy()[:, 1], prediction_train.cpu().detach().numpy()[:, 1]))
        print(f'train-R2: RWP：{r2_train_1},RSP {r2_train_2}, train-rmse: RWP：{rmse_train1},RSP {rmse_train2}\n')
    # 计算测试集上的R²
    for j in test_loader:
        prediction_test = net(j)
        r2_test_1 = r2_score(j.y.cpu().detach().numpy()[:, 0], prediction_test.cpu().detach().numpy()[:, 0])
        r2_test_2 = r2_score(j.y.cpu().detach().numpy()[:, 1], prediction_test.cpu().detach().numpy()[:, 1])
        rmse_test1 = np.sqrt(
            mean_squared_error(j.y.cpu().detach().numpy()[:, 0], prediction_test.cpu().detach().numpy()[:, 0]))
        rmse_test2 = np.sqrt(
            mean_squared_error(j.y.cpu().detach().numpy()[:, 1], prediction_test.cpu().detach().numpy()[:, 1]))
        print(f'test: RWP：{r2_test_1},RSP {r2_test_2}, test-rmse: RWP：{rmse_test1},RSP {rmse_test2}\n')
    # 计算验证集上的R²
if __name__ == '__main__':
    starttime = datetime.datetime.now()
    seed = int(sys.argv[1])
    epoch = int(sys.argv[2])
    lr = float(sys.argv[3])
    dropout_rate = float(sys.argv[4])
    f1 = int(sys.argv[5])
    nlayers = int(sys.argv[6])
    batch_size = int(sys.argv[7])
    path = str(sys.argv[8])
    train(seed=seed, epoch=epoch, lr=lr, dropout_rate=dropout_rate, f1=f1, nlayers=nlayers, batch_size=batch_size,
          path=path)
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
