import torch
import torch.nn as nn
from model import gnnmodel, CrossModel


class AttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        self.linear1 = nn.Linear(input_dim * 2, input_dim)
        self.attention = nn.Linear(input_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        combined_features = torch.cat((x1, x2), dim=1)
        attention_scores = self.attention(self.relu(self.linear1(combined_features)))
        attention_weights = self.sigmoid(attention_scores)
        fused_representation = attention_weights * x1 + (1 - attention_weights) * x2
        return fused_representation


class multimodal_transformer(nn.Module):
    def __init__(self, f1, dropout_rate):
        super(multimodal_transformer, self).__init__()
        dropout_rate = dropout_rate
        f1 = f1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnnmodel = gnnmodel(f1=f1, dropout_rate=dropout_rate)
        self.tabularmodel = CrossModel(f1=f1, dropout_rate=dropout_rate)
        self.attentionfusion = AttentionFusion(input_dim=f1)
        self.f1 = nn.Linear(f1, f1)
        self.f2 = nn.Linear(f1, f1)
        self.f3 = nn.Linear(f1, 2)
        self.drop = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, data):
        data = data.to(self.device)
        gnn_x = self.gnnmodel(data)
        tabular_x = self.tabularmodel(data.tb)
        x = self.attentionfusion(x1=gnn_x, x2=tabular_x)
        x = self.f1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.f2(x)
        x = self.relu(x)
        x = self.f3(x)
        return x
