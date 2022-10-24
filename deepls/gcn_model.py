import torch
from torch import nn
from deepls.gcn_layers import ResidualGatedGCNLayer


class ResidualGatedGCNModel(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.

    References:
        Paper: https://arxiv.org/pdf/1711.07553v2.pdf
        Code: https://github.com/xbresson/spatial_graph_convnets
    """

    def __init__(self, config):
        super(ResidualGatedGCNModel, self).__init__()
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
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.Sequential(*gcn_layers)

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
        x = self.nodes_coord_embedding(x_nodes_coord)  # B x V x H
        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # B x V x V x H
        e_tags = None
        for c in range(self.num_edge_cat_features):
            if e_tags is None:
                e_tags = self.edges_embedding[c](x_edges[..., c])  # B x V x V x H
            else:
                e_tags += self.edges_embedding[c](x_edges[..., c])  # B x V x V x H
        e = torch.cat((e_vals, e_tags), dim=3)
        x, e = self.gcn_layers((x, e))  # B x V x H, B x V x V x H

        return x, e

