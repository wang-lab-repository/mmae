import torch
import torch.nn as nn
import pennylane as qml
from get_data import get_categorical_feature_index


def data_preprocess(x):
    x_num_dim, x_cat_dim, x_cat_cardinalities = get_categorical_feature_index(x, threshold=5)
    x_cat_dim = [i for i in range(len(x_cat_dim))]
    x_num_dim = [i for i in range(len(x_cat_dim), len(x_cat_dim) + len(x_num_dim))]
    if len(x_cat_cardinalities) == 0:
        x_cat_cardinalities = None
    return x, x_num_dim, x_cat_dim, x_cat_cardinalities


def circuit(inputs, weights, n_qubits, n_layers):
    # 编码模块
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        qml.RY(inputs[qub], wires=qub)

    # 纠缠模块
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.CRZ(weights[2 * layer * n_qubits + i], wires=[i, (i + 1) % n_qubits])
        for j in range(n_qubits, 2 * n_qubits):
            qml.RY(weights[2 * layer * n_qubits + j], wires=j % n_qubits)

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


dev1 = qml.device('default.qubit', wires=8)


@qml.qnode(dev1, interface='torch', diff_method='backprop')  #
def qnn1(weights, inputs):
    return circuit(inputs, weights, 8, 1)


dev2 = qml.device('default.qubit', wires=8)


@qml.qnode(dev2, interface='torch', diff_method='backprop')  # , interface='torch', diff_method='backprop'
def qnn2(weights, inputs):
    return circuit(inputs, weights, 8, 2)


dev3 = qml.device('default.qubit', wires=8)


@qml.qnode(dev3, interface='torch',
           diff_method='backprop')
def qnn3(weights, inputs):
    return circuit(inputs, weights, 8, 3)


# def state_prepare(input, qbits):
#     qml.templates.AngleEmbedding(input, wires=range(qbits), rotation="X")
# def qlinear(b1, b2, b3, b4, params):
#     qml.RZ(-torch.pi / 2, wires=b1)
#     qml.RZ(-torch.pi / 2, wires=b2)
#     qml.RZ(params[0], wires=b3)
#     qml.RZ(params[1], wires=b4)
#     qml.RY(params[2], wires=b1)
#     qml.RY(params[3], wires=b2)
#     qml.RY(torch.pi / 2, wires=b3)
#     qml.RY(torch.pi / 2, wires=b4)
#     qml.CNOT([b1, b2])
#     qml.CNOT([b2, b3])
#     qml.CNOT([b3, b4])
#     qml.CNOT([b4, b1])
#
# dev1 = qml.device('default.qubit', wires=4)
#
# @qml.qnode(dev1, interface='torch', diff_method='backprop')  #
# def qnn1(weights, inputs):
#     state_prepare(inputs, 4)
#     qlinear(0, 1, 2, 3, weights[0:4])
#     return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]
#
#
# dev2 = qml.device('default.qubit', wires=4)
#
#
# @qml.qnode(dev2, interface='torch', diff_method='backprop')  # , interface='torch', diff_method='backprop'
# def qnn2(weights, inputs):
#     state_prepare(inputs, 4)
#     qlinear(0, 1, 2, 3, weights[0:4])
#     qlinear(0, 1, 2, 3, weights[4:8])
#     return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]
#
#
# dev3 = qml.device('default.qubit', wires=4)
#
#
# @qml.qnode(dev3, interface='torch',
#            diff_method='backprop')  # ,interface='torch', diff_method='backprop', diff_method='backprop'
# def qnn3(weights, inputs):
#     state_prepare(inputs, 4)
#     qlinear(0, 1, 2, 3, weights[0:4])
#     qlinear(0, 1, 2, 3, weights[4:8])
#     qlinear(0, 1, 2, 3, weights[8:12])
#     return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]


from q_module import MultiheadAttention
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class q_gnnmodel(nn.Module):
    def __init__(self, f1, dropout_rate):
        super(q_gnnmodel, self).__init__()
        dropout_rate = dropout_rate
        f1 = f1
        self.conv1 = GCNConv(9, f1)
        self.conv2 = GCNConv(f1, f1)
        self.conv3 = GCNConv(f1, 8)
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


class QuCrossModel(nn.Module):
    def __init__(self, f1, dropout_rate, nlayers):
        super(QuCrossModel, self).__init__()
        f1 = f1
        dropout_rate = dropout_rate
        self.multiheadattention1 = MultiheadAttention(num_hidden_k=8)
        self.multiheadattention2 = MultiheadAttention(num_hidden_k=8)
        if nlayers == 1:
            self.qcn1 = qml.qnn.TorchLayer(qnn1, weight_shapes={'weights': 16})
        elif nlayers == 2:
            self.qcn1 = qml.qnn.TorchLayer(qnn2, weight_shapes={'weights': 32})
        elif nlayers == 3:
            self.qcn1 = qml.qnn.TorchLayer(qnn3, weight_shapes={'weights': 48})
        if nlayers == 1:
            self.qcn2 = qml.qnn.TorchLayer(qnn1, weight_shapes={'weights': 16})
        elif nlayers == 2:
            self.qcn2 = qml.qnn.TorchLayer(qnn2, weight_shapes={'weights': 32})
        elif nlayers == 3:
            self.qcn2 = qml.qnn.TorchLayer(qnn3, weight_shapes={'weights': 48})
        if nlayers == 1:
            self.qcn3 = qml.qnn.TorchLayer(qnn1, weight_shapes={'weights': 16})
        elif nlayers == 2:
            self.qcn3 = qml.qnn.TorchLayer(qnn2, weight_shapes={'weights': 32})
        elif nlayers == 3:
            self.qcn3 = qml.qnn.TorchLayer(qnn3, weight_shapes={'weights': 48})
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.weight3 = nn.Parameter(torch.randn(1))
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout_rate)
        self.k1 = nn.Linear(8, 8)
        self.q1 = nn.Linear(8, 8)
        self.v1 = nn.Linear(8, 8)
        self.k2 = nn.Linear(8, 8)
        self.q2 = nn.Linear(8, 8)
        self.v2 = nn.Linear(8, 8)

    def forward(self, x):

        at = x.clone()
        x = self.qcn1(x)

        kat = self.k1(at)
        qat = self.q1(at)
        vat = self.v1(at)
        kat = kat.unsqueeze(1)
        qat = qat.unsqueeze(1)
        vat = vat.unsqueeze(1)
        at, attn = self.multiheadattention1(key=kat.clone(), query=qat.clone(), value=vat.clone())
        at = at.squeeze(1)
        x1 = self.weight1 * x + at * (1 - self.weight1)
        x2 = self.weight2 * x + at * (1 - self.weight2)

        at1 = x2.clone()
        x1 = self.qcn2(x1)
        kat1 = self.k2(at1)
        qat1 = self.q2(at1)
        vat1 = self.v2(at1)
        kat1 = kat1.unsqueeze(1)
        qat1 = qat1.unsqueeze(1)
        vat1 = vat1.unsqueeze(1)
        at1, attn1 = self.multiheadattention2(key=kat1.clone(), query=qat1.clone(), value=vat1.clone())
        at1 = at1.squeeze(1)

        xr = x1 * self.weight3 + at1 * (1 - self.weight3)

        xr = self.qcn3(xr)

        return xr

# if __name__ == '__main__':
#     input = torch.randn(666, 16)
#     t = torch.randint(0, 4, (666, 1))
#     x = torch.cat([t, input], dim=1)
#     A = tranmodel()
#     output = A.forward(x)
