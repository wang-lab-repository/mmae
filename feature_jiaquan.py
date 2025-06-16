import torch
import torch.nn as nn
from model import gnnmodel, CrossModel


# 网络结构
class feature_jiaquan(nn.Module):
    def __init__(self, f1, dropout_rate):
        super(feature_jiaquan, self).__init__()
        dropout_rate = dropout_rate
        f1 = f1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnnmodel = gnnmodel(f1=f1, dropout_rate=dropout_rate)
        self.tabularmodel = CrossModel(f1=f1, dropout_rate=dropout_rate)
        self.f1 = nn.Linear(f1, f1)
        self.f2 = nn.Linear(f1, f1)
        self.f3 = nn.Linear(f1, 2)
        self.drop = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.weight = nn.Parameter(torch.randn(1))

    def forward(self, data):
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
