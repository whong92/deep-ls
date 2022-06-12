import pytest
from deepls.gcn_model import TSPRGCNActionNet, model_input_from_states, sample_tour_logit, mask_tour_logit
from tests.test_env import _make_random_state
from deepls.TSP2OptEnv import TSP2OptState
import torch

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
        states.append(TSP2OptState(nodes, edge_weights, rand_tour))
    x_edges, x_edges_values, x_nodes_coord, x_tour = model_input_from_states(states)

    batch_size, num_nodes, _ = x_edges.shape
    edges_sampled, pis, _ = net(x_edges, x_edges_values, x_nodes_coord, x_tour)
    assert edges_sampled.shape == torch.Size([batch_size, 2, 3])
    assert pis.shape == torch.Size([batch_size, 2])

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
        states.append(TSP2OptState(nodes, edge_weights, rand_tour))
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

    # compute logits for tour again, masking out sampled edges, and sample another edge
    tour_logits_1 = mask_tour_logit(tour_logits, actions)
    actions_1, edges_1, pis_1 = sample_tour_logit(tour_logits_1, tour_indices)
    # make sure the sampled logits are equivalent to evaluating only on the logits of previously unsampled actions
    for state, tour_logit, action, action_1, pi, edge, edge_1 in \
            zip(states, tour_logits_1, actions, actions_1, pis_1, edges, edges_1):
        # make sure sampled edges are different
        assert not torch.all(edge[1:] == edge_1[1:])
        assert not torch.all(torch.flip(edge[1:], dims=(0,)) == edge_1[1:])
        # make sure sampled edges exist in tour
        assert state.tour_adj[edge_1[1], edge_1[2]] == 1.
        assert state.tour_adj[edge_1[2], edge_1[1]] == 1.
        # make sure computed prob is the same as if computed on logits with action removed
        if action_1 > action:
            action_1 -= 1
        tour_logit_action_removed = torch.cat((tour_logit[:action], tour_logit[action+1:]), dim=0)
        pi_expected = torch.log_softmax(tour_logit_action_removed, dim=0)[action_1]
        assert torch.isclose(pi, pi_expected)