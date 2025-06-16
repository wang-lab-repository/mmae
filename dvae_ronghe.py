import torch
import torch.nn as nn
import torch.optim as optim

# 设置超参数
batch_size = 64
dim = 10  # 输入数据的维度
dim1 = 5  # 特征提取后的维度
latent_dim = 2  # 潜在空间的维度
beta = 4  # 解纠缠正则化的强度


# 编码器模型
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        return z_mean, z_logvar


# 解码器模型
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc_output = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc_output(x))


# β-VAE模型
class BetaVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(BetaVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        epsilon = torch.randn_like(z_mean)
        z = z_mean + torch.exp(0.5 * z_logvar) * epsilon
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_logvar, epsilon


# 实例化模型
encoder = Encoder(dim, latent_dim)
decoder = Decoder(latent_dim, dim)
beta_vae = BetaVAE(encoder, decoder)

# 设置优化器
optimizer = optim.Adam(beta_vae.parameters(), lr=1e-3)


# 损失函数
# # mse = torch.nn.MSELoss()
# #
# #
# # class MyLoss(torch.nn.Module):
# #     def __init__(self):
# #         super(MyLoss, self).__init__()
# #
# #     def forward(self, y_true, y_pred):
# #         # 自定义损失函数的计算逻辑
# #         scalar = 0.8 * mse(y_true[:, 0], y_pred[:, 0]) + 1.2 * mse(y_true[:, 1], y_pred[:, 1])
# #         return scalar  # 返回一个标量值
def vae_loss(reconstructed, original, z_mean, z_logvar, beta):
    reconstruction_loss = nn.MSELoss()(original, reconstructed)
    # reconstruction_loss = nn.functional.binary_cross_entropy(reconstructed, original, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return reconstruction_loss + beta * kl_divergence
