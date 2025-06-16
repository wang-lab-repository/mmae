import torch.nn as nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# 编码器模型
# 构建基于注意力的自动编码器模型
class AttentionAutoencoder(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim):
        super(AttentionAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softmax(dim=1)
        )
        # 特征提取层
        self.feature_extractor = nn.Linear(hidden_dim, feature_dim)
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # 编码

        encoded = self.encoder(x)

        # 计算注意力权重
        attention_weights = self.attention(encoded)
        # 应用注意力权重
        attended_encoded = attention_weights * encoded
        # 提取特征
        features = self.feature_extractor(attended_encoded)
        # 解码
        reconstructed = self.decoder(features)
        return reconstructed, attention_weights, features
