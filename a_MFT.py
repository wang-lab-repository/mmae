from model import gnnmodel, CrossModel
from a_MFT_ronghe import TransformerEncoder
import torch
import torch.nn as nn

# 网络结构
class mft(nn.Module):
    def __init__(self, f1, dropout_rate):
        super(mft, self).__init__()
        dropout_rate = dropout_rate
        f1 = f1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnnmodel = gnnmodel(f1=f1, dropout_rate=dropout_rate)
        self.tabularmodel = CrossModel(f1=f1, dropout_rate=dropout_rate)

        self.ronghe = TransformerEncoder(2 * f1)

        self.f1 = nn.Linear(2 * f1, 2 * f1)
        self.f2 = nn.Linear(2 * f1, 2 * f1)
        self.f3 = nn.Linear(2 * f1, 2)

        self.drop = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.weight = nn.Parameter(torch.randn(1))  # 定义一个可学习的权重参数

    def forward(self, data):  # 输入：9个特征
        data = data.to(self.device)
        gnn_x = self.gnnmodel(data)
        tabular_x = self.tabularmodel(data.tb)
        # print(gnn_x.shape, tabular_x.shape)
        x = torch.cat([gnn_x, tabular_x], axis=1)  # (batch, 2*f1)
        x = x.unsqueeze(1)  # (batch, 1, 2*f1)
        # print(x.shape)
        x = self.ronghe(x)
        # print(x.shape)
        x = self.f1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.f2(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.f3(x)  # (2432, 2)

        return x