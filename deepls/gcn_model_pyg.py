from functools import lru_cache

from torch_geometric.nn import Sequential as SequentialPyG
from torch_geometric.data import DataLoader, Data
import torch
from torch import nn

from deepls.gcn_model import ResidualGatedGCNModel
from deepls.gcn_layers_pyg import ResidualGatedGCNLayer as ResidualGatedGCNLayerPyG


cache = lru_cache(None)


class ResidualGatedGCNModelPyG(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.

    References:
        Paper: https://arxiv.org/pdf/1711.07553v2.pdf
        Code: https://github.com/xbresson/spatial_graph_convnets
    """

    def __init__(self, config):
        super(ResidualGatedGCNModelPyG, self).__init__()
        # default device
        self.device = 'cpu'
        # Define net parameters
        self.node_dim = config['node_dim']
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = 1  # config['voc_edges_out']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.mlp_layers = config['mlp_layers']
        self.aggregation = config['aggregation']
        self.num_edge_cat_features = config['num_edge_cat_features']
        # Node and edge embedding layers/lookups
        self.nodes_coord_embedding = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        # TODO: make this handle groups of continuous and discrete edge / node features
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim//2, bias=False)
        self.edges_embedding = nn.ModuleList([
            nn.Embedding(self.voc_edges_in, self.hidden_dim//2)
            for _ in range(self.num_edge_cat_features)
        ])
        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append((
                ResidualGatedGCNLayerPyG(self.hidden_dim, self.aggregation),
                'x, edge_index, edge_attr -> x, edge_attr'
            ))
        self.gcn_layers = SequentialPyG('x, edge_index, edge_attr', gcn_layers)
        self.pyg_dl = DataLoader(dataset=[Data(x=torch.zeros(2), edge_index=self.get_edge_index(2))])

    @cache
    def get_edge_index(self, n) -> torch.Tensor:
        edge_index = torch.nonzero(torch.ones(n, n)).T
        # swap edge_index from i,j -> j,i
        edge_index = edge_index[[1, 0], :]
        return edge_index

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device=device, **kwargs)

    def forward(self, x_edges, x_edges_values, x_nodes_coord):
        """
        Args:
            x_edges: Input edge categorical features matrix (batch_size, num_nodes, num_nodes, num_features)
            x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
            x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim)

        Returns:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
            # y_pred_nodes: Predictions for nodes (batch_size, num_nodes)
            loss: Value of loss function
        """
        # Node and edge embedding
        B, V, _ = x_edges_values.shape
        x = self.nodes_coord_embedding(x_nodes_coord)  # B x V x H
        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # B x V x V x H
        e_tags = None
        for c in range(self.num_edge_cat_features):
            if e_tags is None:
                e_tags = self.edges_embedding[c](x_edges[..., c])  # B x V x V x H
            else:
                e_tags += self.edges_embedding[c](x_edges[..., c])  # B x V x V x H
        e = torch.cat((e_vals, e_tags), dim=3)
        # flatten into pyg format to use pyg modules
        eflat = e.view(B * V * V, -1)
        xflat = x.view(B * V, -1)
        edge_index = self.get_edge_index(V)
        edge_index = self.pyg_dl.collate_fn(
            [Data(x=torch.ones(V), edge_index=edge_index)] * B
        ).edge_index.to(self.device)
        x_out, e_out = self.gcn_layers(x=xflat, edge_index=edge_index, edge_attr=eflat)
        # reshape
        e_out = e_out.view(B, V, V, -1)
        x_out = x_out.view(B, V, -1)

        return x_out, e_out


if __name__ == "__main__":
    N = 2
    h = 4

    model_config = {
        "voc_edges_in": 3,
        "hidden_dim": h,
        "num_layers": 5,
        "mlp_layers": 3,
        "aggregation": "mean",
        "node_dim": 2,
        'dont_optimize_policy_steps': 0,
        'value_net_type': 'normal',
        'num_edge_cat_features': 2
    }

    edge_cat_attr = torch.zeros(5, N, N, 2, dtype=torch.int)
    edge_values = torch.zeros(5, N, N, dtype=torch.float)
    x = torch.randn(5, N, 2, dtype=torch.float)

    model = ResidualGatedGCNModelPyG(config=model_config)
    out = model(edge_cat_attr, edge_values, x)
    print(out)

    model = ResidualGatedGCNModel(config=model_config)
    out = model(edge_cat_attr, edge_values, x)
    print(out)
