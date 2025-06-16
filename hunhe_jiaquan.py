import torch
import torch.nn as nn
from model import gnnmodel, CrossModel


# 网络结构
class hunhe_jiaquan(nn.Module):
    def __init__(self, f1, dropout_rate):
        super(hunhe_jiaquan, self).__init__()
        dropout_rate = dropout_rate
        f1 = f1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnnmodel = gnnmodel(f1=f1, dropout_rate=dropout_rate)
        self.tabularmodel = CrossModel(f1=f1, dropout_rate=dropout_rate)

        self.gnn_f1 = nn.Linear(f1, f1)
        self.gnn_f2 = nn.Linear(f1, f1)
        self.gnn_f3 = nn.Linear(f1, 2)
        self.tabular_f1 = nn.Linear(f1, f1)
        self.tabular_f2 = nn.Linear(f1, f1)
        self.tabular_f3 = nn.Linear(f1, 2)
        self.f1 = nn.Linear(f1, f1)
        self.f2 = nn.Linear(f1, f1)
        self.f3 = nn.Linear(f1, 2)

        self.drop = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.weight = nn.Parameter(torch.randn(1))  # 定义一个可学习的权重参数
        self.weight1 = nn.Parameter(torch.randn(1))  # 定义一个可学习的权重参数

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

        # print(gnn_x.shape, tabular_x.shape)
        gnn_x = self.gnn_f1(gnn_x)
        gnn_x = self.drop(gnn_x)
        gnn_x = self.relu(gnn_x)
        gnn_x = self.gnn_f2(gnn_x)
        gnn_x = self.drop(gnn_x)
        gnn_x = self.relu(gnn_x)
        g = self.gnn_f3(gnn_x)

        tabular_x = self.tabular_f1(tabular_x)
        tabular_x = self.drop(tabular_x)
        tabular_x = self.relu(tabular_x)
        tabular_x = self.tabular_f2(tabular_x)
        tabular_x = self.drop(tabular_x)
        tabular_x = self.relu(tabular_x)
        t = self.tabular_f3(tabular_x)

        out = self.weight1 * g + (1 - self.weight1) * t + x

        return out