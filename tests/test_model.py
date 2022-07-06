import pytest
from deepls.gcn_model import TSPRGCNActionNet, model_input_from_states, sample_tour_logit, get_edge_quad_embs
from tests.test_env import _make_random_state
from deepls.TSP2OptEnv import TSP2OptState
from deepls.graph_utils import tour_nodes_to_W
import torch
import numpy as np

@pytest.fixture
def dummy_config():
    return {
        "voc_edges_in": 3,
        "hidden_dim": 128,
        "num_layers": 4,
        "mlp_layers": 3,
        "aggregation": "mean",
        "node_dim": 2,
    }


def test_tsp_rgcn(dummy_config):
    net = TSPRGCNActionNet(config=dummy_config)

    states = []
    for _ in range(100):
        nodes, rand_tour, edge_weights = _make_random_state(N=50)
        states.append(TSP2OptState(nodes, edge_weights, rand_tour, opt_tour_len=0))
    x_edges, x_edges_values, x_nodes_coord, x_tour = model_input_from_states(states)
    # TODO: include this in model_input_from_states?
    x_tour_directed = torch.stack([
        torch.as_tensor(tour_nodes_to_W(state.tour_nodes, directed=True))
        for state in states
    ], dim=0)

    batch_size, num_nodes, _ = x_edges.shape
    edges_sampled, pis, _ = net(x_edges, x_edges_values, x_nodes_coord, x_tour, x_tour, x_tour_directed)
    assert edges_sampled.shape == torch.Size([batch_size, 2, 3])
    assert pis.shape == torch.Size([batch_size])

    # check that sampled edges are indeed in the state
    for state, edges in zip(states, edges_sampled):
        edge_0, edge_1 = edges
        # make sure sampled edges are different
        assert not torch.all(edge_0[1:] == edge_1[1:])
        assert not torch.all(torch.flip(edge_0[1:], dims=(0,)) == edge_1[1:])
        # make sure sampled edges exist in tour
        assert state.tour_adj[edge_0[1], edge_0[2]] == 1.
        assert state.tour_adj[edge_0[2], edge_0[1]] == 1.
        assert state.tour_adj[edge_1[1], edge_1[2]] == 1.
        assert state.tour_adj[edge_1[2], edge_1[1]] == 1.


def test_sample_tour_logits(dummy_config):
    """
    tests sample_tour_logit and mask_tour_logit
    :param dummy_config:
    :return:
    """
    states = []
    for _ in range(100):
        nodes, rand_tour, edge_weights = _make_random_state(N=50)
        states.append(TSP2OptState(nodes, edge_weights, rand_tour, opt_tour_len=0))
    x_edges, x_edges_values, x_nodes_coord, x_tour = model_input_from_states(states)
    batch_size, num_nodes, _ = x_edges.shape

    # format tour_indices properly
    tour_indices = torch.nonzero(x_tour)
    tour_indices = tour_indices[tour_indices[:, 1] < tour_indices[:, 2]]
    tour_indices = tour_indices.reshape(batch_size, num_nodes, -1)

    # random_logits
    tour_logits = torch.rand(batch_size, num_nodes)
    # sample
    actions, edges, pis = sample_tour_logit(tour_logits, tour_indices)
    # check
    for state, tour_logit, action, pi, edge in zip(states, tour_logits, actions, pis, edges):
        assert state.tour_adj[edge[1], edge[2]] == 1.
        assert state.tour_adj[edge[2], edge[1]] == 1.
        assert torch.isclose(pi, torch.log_softmax(tour_logit, dim=0)[action])


def compute_deltas_naive(tour_indices_cat, x_tour_directed, x_edges_values):
    deltas = torch.zeros(*tour_indices_cat.shape[:2]) # b x v
    for bidx, tour_indices_cat_b in enumerate(tour_indices_cat):
        for i, (_, u, v, _, x, y) in enumerate(tour_indices_cat_b):
            if x_tour_directed[bidx, u, v] == 0:
                tmp = v
                v = u
                u = tmp
            if x_tour_directed[bidx, x, y] == 0:
                tmp = y
                y = x
                x = tmp
            # IMPORTANT! check that u, v and x, y are in fact in the correct directions
            assert x_tour_directed[bidx, x, y] == 1 and x_tour_directed[bidx, u, v] == 1
            delta = x_edges_values[bidx, u, v] + x_edges_values[bidx, x, y]
            delta = x_edges_values[bidx, u, x] + x_edges_values[bidx, v, y] - delta
            deltas[bidx, i] = delta
    return deltas


def test_get_edge_quad_embs():
    for _ in range(10):
        N = np.random.randint(4, 200)
        nodes, rand_tour, edge_weights = _make_random_state(N)
        state = TSP2OptState(nodes, edge_weights, rand_tour, opt_tour_len=0)
        x_edges, x_edges_values, x_nodes_coord, x_tour = model_input_from_states([state])
        x_tour_directed = torch.as_tensor(tour_nodes_to_W(state.tour_nodes, directed=True)).unsqueeze(0)
        # if we treated the edge values as a 1-d edge embedding, and got the embedding of edge quads corresponding to
        # each 2opt move
        edge_quad_embs, tour_indices_cat = get_edge_quad_embs(x_edges_values.unsqueeze(3), x_tour, x_tour_directed)
        # and we use a naive calculation of the delta of each move
        deltas_stupid = compute_deltas_naive(
            tour_indices_cat=tour_indices_cat,
            x_tour_directed=x_tour_directed,
            x_edges_values=x_edges_values
        )
        # the two should match
        # (emb(u, x) + emb(v, y)) - (emb(u, v) + emb(x, y)) = delta of the 2opt move involving edges (u, v) and (x, y)
        delta_from_embs = \
            -edge_quad_embs[..., 0] + \
            -edge_quad_embs[..., 1] + \
            edge_quad_embs[..., 2] + \
            edge_quad_embs[..., 3]
        assert torch.allclose(delta_from_embs, deltas_stupid, atol=1e-06)
