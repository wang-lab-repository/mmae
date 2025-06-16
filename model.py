import torch
import torch.nn as nn
import math
import typing as ty
import torch.nn.init as nn_init
from torch import Tensor
from get_data import get_categorical_feature_index
from module import MultiheadAttention
from torch_geometric.nn import GCNConv, TransformerConv, SGConv
from torch_geometric.nn import global_mean_pool


class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
            self,
            d_numerical: int,
            categories: ty.Optional[ty.List[int]],
            d_token: int,
            bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


def data_preprocess(x):
    x_num_dim, x_cat_dim, x_cat_cardinalities = get_categorical_feature_index(x, threshold=5)
    x_cat_dim = [i for i in range(len(x_cat_dim))]
    x_num_dim = [i for i in range(len(x_cat_dim), len(x_cat_dim) + len(x_num_dim))]
    if len(x_cat_cardinalities) == 0:
        x_cat_cardinalities = None
    return x, x_num_dim, x_cat_dim, x_cat_cardinalities


# 网络结构
class gnnmodel(nn.Module):
    def __init__(self, f1, dropout_rate):
        super(gnnmodel, self).__init__()
        dropout_rate = dropout_rate
        f1 = f1
        self.conv1 = GCNConv(9, f1)
        self.conv2 = GCNConv(f1, f1)
        self.conv3 = GCNConv(f1, f1)
        self.drop = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)

        return x


class CrossModel(nn.Module):
    def __init__(self, f1, dropout_rate):
        super(CrossModel, self).__init__()
        f1 = f1
        dropout_rate = dropout_rate
        self.multiheadattention1 = MultiheadAttention(num_hidden_k=34)
        self.multiheadattention2 = MultiheadAttention(num_hidden_k=f1)
        self.fc11 = nn.Linear(34, f1)
        self.fc12 = nn.Linear(f1, f1)
        self.fc13 = nn.Linear(f1, 34)
        self.fc21 = nn.Linear(34, f1)
        self.fc22 = nn.Linear(f1, f1)
        self.fc23 = nn.Linear(f1, 34)
        self.fc1 = nn.Linear(34, f1)
        self.fc2 = nn.Linear(f1, f1)
        self.fc3 = nn.Linear(f1, f1)
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.weight3 = nn.Parameter(torch.randn(1))
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout_rate)
        self.k1 = nn.Linear(34, 34)
        self.q1 = nn.Linear(34, 34)
        self.v1 = nn.Linear(34, 34)
        self.k2 = nn.Linear(34, 34)
        self.q2 = nn.Linear(34, 34)
        self.v2 = nn.Linear(34, 34)

    def forward(self, x):

        at = x.clone() # 经过深度复制之后的

        x = self.fc11(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc12(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc13(x)

        kat = self.k1(at)
        qat = self.q1(at)
        vat = self.v1(at)
        kat = kat.unsqueeze(1)
        qat = qat.unsqueeze(1)
        vat = vat.unsqueeze(1)
        at, attn = self.multiheadattention1(key=kat.clone(), query=qat.clone(), value=vat.clone())
        at = at.squeeze(1)

        x1 = x * self.weight1 + at * (1 - self.weight1)
        x2 = x * self.weight2 + at * (1 - self.weight2)

        at1 = x2.clone()
        x1 = self.fc21(x1)
        x1 = self.relu(x1)
        x1 = self.drop(x1)
        x1 = self.fc22(x1)
        x1 = self.relu(x1)
        x1 = self.drop(x1)
        x1 = self.fc23(x1)
        kat1 = self.k2(at1)
        qat1 = self.q2(at1)
        vat1 = self.v2(at1)
        kat1 = kat1.unsqueeze(1)
        qat1 = qat1.unsqueeze(1)
        vat1 = vat1.unsqueeze(1)
        at1, attn1 = self.multiheadattention2(key=kat1.clone(), query=qat1.clone(), value=vat1.clone())
        at1 = at1.squeeze(1)

        xr = x1 * self.weight3 + at1 * (1 - self.weight3)
        xr = self.fc1(xr)
        xr = self.relu(xr)
        xr = self.drop(xr)
        xr = self.fc2(xr)
        xr = self.relu(xr)
        xr = self.drop(xr)
        xr = self.fc3(xr)
        return xr

# if __name__ == '__main__':
#     input = torch.randn(666, 16)
#     t = torch.randint(0, 4, (666, 1))
#     x = torch.cat([t, input], dim=1)
#     A = tranmodel()
#     output = A.forward(x)
