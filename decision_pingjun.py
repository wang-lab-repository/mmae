import torch
import torch.nn as nn
from model import gnnmodel, CrossModel


class decision_pingjun(nn.Module):
    def __init__(self, f1, dropout_rate):
        super(decision_pingjun, self).__init__()
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

        self.drop = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, data):
        data = data.to(self.device)
        gnn_x = self.gnnmodel(data)
        tabular_x = self.tabularmodel(data.tb)
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

        out = (g + t) / 2

        return out
