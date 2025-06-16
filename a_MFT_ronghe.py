import sys
from torch.nn import LayerNorm, Linear, Dropout, Softmax
import copy
import torch.nn as nn
import torch


class MCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        self.scale = qk_scale or dim ** -0.5
        self.dim = dim
        self.wq = nn.Linear(dim, dim * num_heads, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim * num_heads, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim * num_heads, bias=qkv_bias)
        #         self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # print(1)
        # print(x.shape) (b,1,c)
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, self.dim).permute(0, 2, 1,
                                                                                    3)  # B1C -> B1H(C/H) -> BH1(C/H)
        # print(q.shape) (b,head,1,c)
        k = self.wk(x).reshape(B, N, self.num_heads, self.dim).permute(0, 2, 1,
                                                                       3)  # BNC -> BNH(C/H) -> BHN(C/H)
        # print(k.shape) (b,head,1,c)
        v = self.wv(x).reshape(B, N, self.num_heads, self.dim).permute(0, 2, 1,
                                                                       3)  # BNC -> BNH(C/H) -> BHN(C/H)
        # print(v.shape) (b,head,1,c)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        #         attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        #         attn = self.attn_drop(attn)
        x = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2)
        #         x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, 1, C * self.num_heads)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim):
        super(Mlp, self).__init__()
        self.fc1 = Linear(dim, 512)
        self.fc2 = Linear(512, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, dim):
        super(Block, self).__init__()
        self.hidden_size = dim
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        self.ffn_norm = LayerNorm(dim, eps=1e-6)
        self.ffn = Mlp(dim)
        #         self.attn = Attention(dim = 64)
        self.attn = MCrossAttention(dim=dim)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for _ in range(2):
            layer = Block(dim)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        for layer_block in self.layer:
            x = layer_block(x)
            # print(x.shape)

        encoded = self.encoder_norm(x)
        # print(x.shape)
        return encoded[:, 0]
# class TabularCrossAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         B, C = x.shape
#         qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # Make torchscript happy (cannot use tensor as tuple)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         # v 的形状应该是 [B, num_heads, C // num_heads]
#         attn_v = attn @ v
#         print(f"attn_v shape before transpose: {attn_v.shape}")
#
#         attn_v = attn_v.transpose(1, 2).contiguous()  # 确保内存连续
#         print(f"attn_v shape after transpose: {attn_v.shape}")
#
#         # 在此检查 attn_v 的总元素数是否与目标形状匹配
#         target_elements = attn_v.shape[0] * attn_v.shape[1] * attn_v.shape[2]
#         print(f"Target elements: {target_elements}")
#
#         # 确保重塑的形状与元素数量匹配
#         assert target_elements == B * C, "Reshape shape does not match the number of elements."
#
#         x = attn_v.view(B, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#
# class Mlp(nn.Module):
#     def __init__(self, dim):
#         super(Mlp, self).__init__()
#         self.fc1 = Linear(dim, 512)
#         self.fc2 = Linear(512, dim)
#         self.act_fn = nn.GELU()
#         self.dropout = Dropout(0.1)
#
#         self._init_weights()
#
#     def _init_weights(self):
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.normal_(self.fc1.bias, std=1e-6)
#         nn.init.normal_(self.fc2.bias, std=1e-6)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act_fn(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x
#
#
# class TabularBlock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.hidden_size = dim
#         self.attention_norm = nn.LayerNorm(dim, eps=1e-6)
#         self.ffn_norm = nn.LayerNorm(dim, eps=1e-6)
#         self.ffn = Mlp(dim)
#         self.attn = TabularCrossAttention(dim=dim)
#
#     def forward(self, x):
#         h = x
#         x = self.attention_norm(x)
#         x = self.attn(x)
#         x = x + h
#
#         h = x
#         x = self.ffn_norm(x)
#         x = self.ffn(x)
#         x = x + h
#         return x
#
#
# class TabularTransformerEncoder(nn.Module):
#     def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
#                  drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
#         super().__init__()
#         self.layer = nn.ModuleList()
#         self.encoder_norm = norm_layer(dim, eps=1e-6)
#         for _ in range(2):  # Number of layers
#             layer = TabularBlock(dim)
#             self.layer.append(copy.deepcopy(layer))
#
#     def forward(self, x):
#         for layer_block in self.layer:
#             x = layer_block(x)
#         encoded = self.encoder_norm(x)
#         return encoded
