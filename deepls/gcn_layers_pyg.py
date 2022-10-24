import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing


from torch import Tensor
from torch.nn import Linear


class ResidualGatedGCNLayer(MessagePassing):

    def __init__(self, hidden_dim: int, aggregation='sum'):
        super().__init__(agg='add')
        self.aggregation = aggregation
        self.h = hidden_dim

        self.Uxx = Linear(hidden_dim, hidden_dim)
        self.Vxx = Linear(hidden_dim, hidden_dim)
        self.Uee = Linear(hidden_dim, hidden_dim)
        self.Vxe = Linear(hidden_dim, hidden_dim)

        self.bn_node = nn.BatchNorm1d(hidden_dim, track_running_stats=True)
        self.bn_edge = nn.BatchNorm1d(hidden_dim, track_running_stats=True)

    def forward(
        self,
        x: Tensor,
        edge_index,
        edge_attr: Tensor = None
    ):
        Uxx = self.Uxx(x)
        Vxx = self.Vxx(x)
        Vxe = self.Vxe(x)

        e_out = self.edge_updater(Vxe=Vxe, edge_attr=edge_attr, edge_index=edge_index)

        e_gate = torch.sigmoid(e_out)
        x_out = self.propagate(edge_index=edge_index, e_gate=e_gate, Vxx=Vxx, Uxx=Uxx)

        e_out = torch.relu(self.bn_edge(e_out))
        x_out = torch.relu(self.bn_node(x_out))

        e_out = edge_attr + e_out
        x_out = x + x_out

        return x_out, e_out

    def update(self, aggr_out, Uxx) -> Tensor:
        if self.aggregation == 'mean':
            x_out, gate_sum = aggr_out[:, :self.h], aggr_out[:, self.h:]
            x_out = x_out / gate_sum
        else:
            x_out = aggr_out
        return Uxx + x_out

    def edge_update(self, Vxe_i, Vxe_j, edge_attr) -> Tensor:
        # edge_attr is E x h, x is N x h
        Uee = self.Uee(edge_attr)  # E x h
        e_out = Uee + Vxe_i + Vxe_j
        return e_out

    def message(self, e_gate, Vxx_j) -> Tensor:
        # message consists of
        if self.aggregation == 'sum':
            return e_gate * Vxx_j
        elif self.aggregation == 'mean':
            return torch.cat((e_gate * Vxx_j, e_gate), dim=1)
        else:
            raise ValueError("aggregation not recognized")


class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        layers = []
        for layer in range(self.L + 1):
            layer_in_dim = input_dim if layer == 0 else hidden_dim
            layer_out_dim = hidden_dim if layer < self.L else output_dim
            layers.append(nn.Linear(layer_in_dim, layer_out_dim, True))
            if layer < self.L:
                layers.append(nn.ReLU())
        self.U = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)

        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        y = self.U(x)
        return y


# testing the pyg implementation
# from deepls.gcn_layers import ResidualGatedGCNLayer as RGCN_og
# from torch_geometric.data import DataLoader
#
#
# if __name__ == "__main__":
#     from torch_geometric.data import Data
#     N = 2
#     h = 4
#     device = 'cuda'
#     edge_index = torch.nonzero(torch.ones(N, N)).T
#     # swap edge_index from i,j -> j,i
#     edge_index = edge_index[[1, 0], :]
#     edge_attr = torch.randn(N, N, h, dtype=torch.float)
#     edge_attr_flat = edge_attr.reshape(N * N, -1)
#     x = torch.randn(N, h, dtype=torch.float)
#
#     print(edge_attr)
#     print(edge_attr[0, 0])
#     print(edge_attr[0, 1])
#     print(edge_attr[1, 0])
#     print(edge_attr[1, 1])
#     print(edge_index)
#     print(edge_attr_flat)
#
#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_flat).to(device)
#     dl = DataLoader(dataset=[data])
#
#     batch = dl.collate_fn([data, data, data])
#     batch.to(device)
#
#     model = ResidualGatedGCNLayer(hidden_dim=h, aggregation='mean')
#     model_og = RGCN_og(hidden_dim=h, aggregation='mean')
#
#     def copy_linear_weights(src: Linear, dst: Linear):
#         dst.weight = torch.nn.Parameter(src.weight.clone())
#         dst.bias = torch.nn.Parameter(src.bias.clone())
#
#     copy_linear_weights(model_og.node_feat.U, model.Uxx)
#     copy_linear_weights(model_og.node_feat.V, model.Vxx)
#     copy_linear_weights(model_og.edge_feat.U, model.Uee)
#     copy_linear_weights(model_og.edge_feat.V, model.Vxe)
#
#     model.to(device)
#     out = model(batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
#
#     model_og.to(device)
#     out_og = model_og((torch.stack([x] * 3, dim=0).to(device), torch.stack([edge_attr] * 3, dim=0).to(device)))
#
#     print(' ================= ')
#     print(out)
#
#     print(' ================= ')
#     print(out_og)
