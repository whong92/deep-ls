import copy
from typing import Optional

import torch
from torch import nn
from torch.distributions import categorical as tdc

from deepls.gcn_layers import MLP
from deepls.gcn_model import ResidualGatedGCNModel
# from deepls.gcn_model_pyg import ResidualGatedGCNModelPyG


def model_input_from_states(states):
    """
    :param states: list of TSP2OptStates
    :return:
    x_edges - b x v x v boolean - adj matrix (in this case it's just all ones since we assume a fully-connected graph)
    x_edges_values -  b x v x v - edge weights (distances)
    x_nodes_coord -  b x v - node coordinates
    x_tour -  b x v x v - adj matrix representing the tour
    """
    x_edges = []
    x_edges_values = []
    x_nodes_coord = []
    x_tour = []
    for state in states:
        x_tour.append(torch.Tensor(state.tour_adj).unsqueeze(0).to(torch.long))
        x_edges.append(torch.ones_like(x_tour[-1]))
        x_edges_values.append(torch.Tensor(state.edge_weights).unsqueeze(0))
        x_nodes_coord.append(torch.Tensor(state.nodes_coord).unsqueeze(0))
    return torch.cat(x_edges, dim=0), \
           torch.cat(x_edges_values, dim=0), \
           torch.cat(x_nodes_coord, dim=0), \
           torch.cat(x_tour, dim=0)


def sample_tour_logit(
    tour_logit: torch.Tensor,
    tour_indices: torch.Tensor,
    actions: Optional[torch.Tensor] = None,
    tau: float = 1.0,
    greedy: bool = False
):
    """
    given edge pairs, sample an action (an edge pair) and return the associated edges, and log prob
    if actions is not None, don't sampled and compute the log prob of the supplied actions
    instead
    :param tour_logit: b x n_edge_pairs
    :param tour_indices: b x n_edge_pairs x 6 - last dimension is (_, u, v, _, x, y) where (u, v) and (x, y) are
    the edges to swap
    :param actions: b - tensor of actions, if not none, we don't sample anything and just compute the log prob
    :param tau: float - temperature for softmax over tour logits - high temperatures encourage exploration
    :param greedy: boolean - greedily choose the best edge ie argmax (tour_logit)
    :return:
    actions: the index of the action in [0, n_edge_pairs - 1] s.t. tour_indices[batch, action] = (_, u, v), (_, x, y)
    edges: the edges (u, v), (x, y) of the actions
    pi: the log prob of the action - for computing policy gradients
    """
    batch_size = tour_logit.shape[0]
    # make a distribution of the tour and sample action
    if actions is None and not greedy:
        tour_dist_sample = tdc.Categorical(logits=tour_logit / tau)
        actions = tour_dist_sample.sample()
    elif actions is None and greedy:
        actions = torch.argmax(tour_logit / tau, dim=1) # b
    tour_dist = tdc.Categorical(logits=tour_logit / tau)
    # calculate log_probs for action
    pi = tour_dist.log_prob(actions)  # b
    # convert actions into edges via tour indices
    edges = tour_indices[torch.arange(batch_size), actions]  # b x 3
    return actions, edges, pi


def normalize_edges_tour(x_tour_directed, bs, us, vs):
    """
    given a directed tour definition, and edges us, vs, with batch indices bs, normalize it
    by flipping around edges that are reversed
    :param x_tour_directed: b x v x v
    :param bs: b * v (flattened)
    :param us: b * v (flattened)
    :param vs: b * v (flattened)
    :return:
    """
    is_flipped = ~(x_tour_directed[bs, us, vs])
    us_right = us.clone()
    vs_right = vs.clone()
    us_right[is_flipped] = vs[is_flipped]
    vs_right[is_flipped] = us[is_flipped]
    assert torch.all(x_tour_directed[bs, us_right, vs_right])
    return us_right, vs_right


def get_edge_quad_embs(e_emb, x_tour, x_tour_directed):
    """
    this method is fairly tedious so excuse the verbosity and repetition

    for every edge pair (u, v), (x, y) in x_tour, identify the 2opt neighbor edges
    (u, x), (v, y), and concatenate the edge embeddings emb(u, v), emb(x, y), emb(u, x),
    emb(v, y) to give the embedding for the 2opt action

    :param e_emb: b x v x v x h - edge embeddings
    :param x_tour: b x v x v - binary tour adjacency mat
    :param x_tour_directed: b x v x v - binary directed tour adjacency mat
    :return:
    action_quad - quadruplet embeddings of each 2opt action, [emb(u, v), emb(x, y), emb(u, x), emb(v, y)] - (b, n_edge_pairs, 4 * emb_size)
    tour_indices_cat - the edges of the 2opt action (u, v), (x, y) - (b, n_edge_pairs, 6)
    """
    b, v, _, _ = e_emb.shape
    tour_indices = torch.nonzero(x_tour)
    tour_indices = tour_indices[tour_indices[:, 1] < tour_indices[:, 2]]
    tour_indices = tour_indices.reshape(b, v, -1)  # b x v x 3 - last dim is (b, u, v)
    tour_indices_cat = torch.cat([
        tour_indices.unsqueeze(2).repeat(1, 1, v, 1),
        tour_indices.unsqueeze(1).repeat(1, v, 1, 1)
    ], dim=3) # b x v x v x 3
    # indices of a v x v matrix where j > i
    rs, cs = torch.triu_indices(v, v, offset=1)
    tour_indices_cat = tour_indices_cat[:, rs, cs, :]  # b x v x 6 - last dim is (b, u, v, b, x, y)

    x_tour_directed = x_tour_directed.bool()
    b, n_edge_pairs, _ = tour_indices_cat.shape
    bs = tour_indices_cat[:, :, 0].reshape(-1)
    us = tour_indices_cat[:, :, 1].reshape(-1)
    vs = tour_indices_cat[:, :, 2].reshape(-1)
    xs = tour_indices_cat[:, :, 4].reshape(-1)
    ys = tour_indices_cat[:, :, 5].reshape(-1)
    # normalize the edges so they always point forward in the tour, this way there is a unique (u, v) and (x, y)
    us, vs = normalize_edges_tour(x_tour_directed, bs, us, vs)
    xs, ys = normalize_edges_tour(x_tour_directed, bs, xs, ys)

    orig_pair_emb = torch.cat([e_emb[bs, us, vs, :], e_emb[bs, xs, ys, :]], dim=1)
    new_pair_emb = torch.cat([e_emb[bs, us, xs, :], e_emb[bs, vs, ys, :]], dim=1)
    # compile [emb(u, v), emb(x, y), emb(u, x), emb(v, y)]
    action_quad = torch.cat((orig_pair_emb, new_pair_emb), dim=1).reshape((b, n_edge_pairs, -1))

    return action_quad, tour_indices_cat


class TSPRGCNValueNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = copy.deepcopy(config)
        config['num_edge_cat_features'] = 2
        # self.rgcn = ResidualGatedGCNModelPyG(config)
        self.rgcn = ResidualGatedGCNModel(config)
        self.hidden_dim = config['hidden_dim']
        self.value_net = torch.nn.Sequential(
            MLP(self.hidden_dim, self.hidden_dim, output_dim=1),
        )

    def forward(self, x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour):
        """
        x_edges: b x v x v
        x_edges_values: b x v x v
        x_nodes_coord: b x v x 2
        x_tour: b x v x v
        """
        x_cat = torch.stack([x_tour, x_best_tour], dim=3)
        x_emb, e_emb = self.rgcn(x_cat, x_edges_values, x_nodes_coord)
        x_emb = self.value_net(x_emb).squeeze(-1) # b x v x 1
        value = torch.mean(x_emb, dim=1)
        return value


class TSPRGCNLogNormalValueNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = copy.deepcopy(config)
        config['num_edge_cat_features'] = 2
        # self.rgcn = ResidualGatedGCNModelPyG(config)
        self.rgcn = ResidualGatedGCNModel(config)
        self.hidden_dim = config['hidden_dim']
        self.value_net = torch.nn.Sequential(
            MLP(self.hidden_dim, self.hidden_dim, output_dim=2),
        )  # mu and sigma

    def forward(self, x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour):
        """
        x_edges: b x v x v
        x_edges_values: b x v x v
        x_nodes_coord: b x v x 2
        x_tour: b x v x v
        """
        x_cat = torch.stack([x_tour, x_best_tour], dim=3)
        x_emb, e_emb = self.rgcn(x_cat, x_edges_values, x_nodes_coord)
        x_emb = self.value_net(x_emb) # b x v x 2
        x_emb_mean = torch.mean(x_emb, dim=1) # b x 2
        mu, sigma = x_emb_mean[:, 0], torch.exp(x_emb_mean[:, 1])
        value_dist = torch.distributions.LogNormal(loc=mu, scale=sigma)
        return value_dist


class TSPRGCNActionNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = copy.deepcopy(config)
        config['num_edge_cat_features'] = 2
        self.rgcn = ResidualGatedGCNModel(config)
        self.hidden_dim = config['hidden_dim']
        self.action_cost_linear = MLP(
            input_dim=1,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            L=0
        )
        self.pre_action_net = MLP(
            input_dim=self.hidden_dim * 5,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            L=0
        )
        self.action_net = MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=1, L=3)
        self.greedy = False

    def compute_state_tour_embeddings(
        self,
        x_edges,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        x_tour_directed
    ):
        x_cat = torch.stack([x_tour, x_best_tour], dim=3)
        # action 0
        x_emb, e_emb = self.rgcn(x_cat, x_edges_values, x_nodes_coord)
        # get embeddings only for edges on the current tour (the ones we want to sample)
        tour_emb_pairs, tour_indices_pairs = get_edge_quad_embs(e_emb, x_tour, x_tour_directed)
        return tour_emb_pairs, tour_indices_pairs

    def set_greedy(self, greedy=False):
        # set greedy decoding
        self.greedy = greedy

    def forward(
        self,
        x_edges,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        x_tour_directed
    ):
        """
        x_edges: b x v x v
        x_edges_values: b x v x v
        x_nodes_coord: b x v x 2
        x_tour: b x v x v
        x_best_tour: b x v x v
        tour_deltas: b x v
        """
        b, _, _ = x_edges.shape
        tour_logits, tour_indices_cat = self.get_tour_logits(
            x_edges,
            x_edges_values,
            x_nodes_coord,
            x_tour,
            x_best_tour,
            x_tour_directed
        )
        actions, edges, pi = sample_tour_logit(tour_logits.squeeze(-1), tour_indices_cat, greedy=self.greedy)

        return edges.reshape(b, 2, 3), pi, actions

    def get_action_pref(
        self,
        x_edges,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        actions,
        x_tour_directed
    ):
        """
        x_edges: b x v x v
        x_edges_values: b x v x v
        x_nodes_coord: b x v x 2
        x_tour: b x v x v
        x_best_tour: b x v x v
        """
        b, _, _ = x_edges.shape
        tour_logits, tour_indices_cat = self.get_tour_logits(
            x_edges,
            x_edges_values,
            x_nodes_coord,
            x_tour,
            x_best_tour,
            x_tour_directed
        )
        actions, edges, pi = sample_tour_logit(tour_logits.squeeze(-1), tour_indices_cat, actions=actions)

        return edges.reshape(b, 2, 3), pi, actions

    def compute_action_cost_emb(
        self,
        x_edges_values,
        x_tour,
        x_tour_directed,
    ):
        edge_quad_values, _ = get_edge_quad_embs(
            x_edges_values.unsqueeze(3),
            x_tour,
            x_tour_directed
        )
        action_cost = \
            edge_quad_values[..., 2] + edge_quad_values[..., 3] \
            - edge_quad_values[..., 0] - edge_quad_values[..., 1]
        action_cost_emb = self.action_cost_linear(action_cost.unsqueeze(2))
        return action_cost_emb

    def get_tour_logits(
        self,
        x_edges,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        x_tour_directed
    ):
        tour_emb_cat, tour_indices_cat = self.compute_state_tour_embeddings(
            x_edges,
            x_edges_values,
            x_nodes_coord,
            x_tour,
            x_best_tour,
            x_tour_directed
        )
        b, n_edge_pairs, _ = tour_emb_cat.shape
        action_cost_emb = self.compute_action_cost_emb(
            x_edges_values,
            x_tour,
            x_tour_directed,
        )
        state_emb = torch.cat((tour_emb_cat, action_cost_emb), dim=2)
        state_emb = self.pre_action_net(state_emb)
        tour_logits = self.action_net(state_emb)
        return tour_logits, tour_indices_cat
