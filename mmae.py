from model import gnnmodel, CrossModel
from a_MFT_ronghe import TransformerEncoder
from att_autoencoder_ronghe import AttentionAutoencoder
import torch
import torch.nn as nn

# 网络结构
class mmae(nn.Module):
    def __init__(self, f1, dropout_rate):
        super(mmae, self).__init__()
        dropout_rate = dropout_rate
        f1 = f1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnnmodel = gnnmodel(f1=f1, dropout_rate=dropout_rate)
        self.tabularmodel = CrossModel(f1=f1, dropout_rate=dropout_rate) # 表格特征提取
        self.ronghe = AttentionAutoencoder(2 * f1, f1, 2 * f1)

        self.attention1 = nn.Sequential(
            nn.Linear(2 * f1, 2 * f1),
            nn.Softmax(dim=1)
        )
        self.attention2 = nn.Sequential(
            nn.Linear(2 * f1, 2 * f1),
            nn.Softmax(dim=1)
        )
        self.attention3 = nn.Sequential(
            nn.Linear(2 * f1, 2 * f1),
            nn.Softmax(dim=1)
        )
        self.ronghe1 = TransformerEncoder(2 * f1)
        self.ronghe2 = TransformerEncoder(2 * f1)
        self.ronghe3 = TransformerEncoder(2 * f1)

        self.f1 = nn.Linear(2 * f1, 2 * f1)
        self.f2 = nn.Linear(2 * f1, 2 * f1)
        self.f3 = nn.Linear(2 * f1, 2)

        self.drop = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.weight = nn.Parameter(torch.randn(1))  # 定义一个可学习的权重参数
        self.weight1 = nn.Parameter(torch.randn(1))  # 定义一个可学习的权重参数
        self.weight2 = nn.Parameter(torch.randn(1))  # 定义一个可学习的权重参数

    def forward(self, data):  # 输入：9个特征
        data = data.to(self.device)
        gnn_x = self.gnnmodel(data)
        tabular_x = self.tabularmodel(data.tb)

        # mmae的模态融合部分
        x = torch.cat([gnn_x, tabular_x], axis=1)  # (batch, 2*f1) 编码器输入

        h = x  # (batch, 2*f1) 编码器输出
        # x——x_fused
        reconstructed, attention_weights, x = self.ronghe(x)

        gt = torch.cat([gnn_x, tabular_x], axis=1)
        gx = torch.cat([gnn_x, x], axis=1)
        tx = torch.cat([tabular_x, x], axis=1)##
        gt_attention_weights = self.attention1(gt)
        gx_attention_weights = self.attention2(gx)
        tx_attention_weights = self.attention3(tx)
        # 应用注意力权重
        gt = (gt_attention_weights * gt).unsqueeze(1)
        gx = (gx_attention_weights * gx).unsqueeze(1)
        tx = (tx_attention_weights * tx).unsqueeze(1)
        gt = self.ronghe1(gt)
        gx = self.ronghe2(gx)
        tx = self.ronghe3(tx)

        x = self.weight1 * gt + self.weight2 * gx + (1 - self.weight1 - self.weight2) * tx

        x = self.f1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.f2(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.f3(x)  # (2432, 2)

        return x, reconstructed, attention_weights, h
