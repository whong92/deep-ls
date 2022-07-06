from deepls.TSP2OptEnv import TSP2OptState
import numpy as np
from scipy.spatial.distance import pdist, squareform
from deepls.graph_utils import is_valid_tour, tour_nodes_to_W, tour_nodes_to_tour_len
from deepls.agent import _make_random_move
import pytest


def _make_random_state(N = 50):
    nodes = np.random.randn(N, 2)
    rand_tour = np.random.permutation(N)
    edge_weights = squareform(pdist(nodes, metric='euclidean'))
    return nodes, rand_tour, edge_weights


@pytest.mark.parametrize('N', [10, 20, 50, 100])
def test_state(N):
    for _ in range(100):
        nodes, rand_tour, edge_weights = _make_random_state(N)
        state = TSP2OptState(nodes, edge_weights, rand_tour, opt_tour_len=0)
        assert is_valid_tour(state.tour_nodes, state.num_nodes)

        # choose 2 random edges
        e1, e2 = _make_random_move(state)
        e_swap = np.vstack((e1, e2)).T
        # if edges overlap, nothing is changed
        overlap = len(set(e1).intersection(set(e2))) != 0
        orig_tour_adj = state.tour_adj.copy()
        orig_tour_nodes = state.tour_nodes.copy()
        # mask to denote which edges have been swapped and will no longer exist
        swapped_tour_adj_mask = np.zeros_like(orig_tour_adj)
        swapped_tour_adj_mask[e_swap[0], e_swap[1]] = 1
        swapped_tour_adj_mask[e_swap[1], e_swap[0]] = 1

        state.apply_move(e1, e2)

        assert np.all(tour_nodes_to_W(state.tour_nodes) == state.tour_adj)
        assert np.isclose(tour_nodes_to_tour_len(state.tour_nodes, edge_weights), state.tour_len)
        if overlap:
            assert np.all(orig_tour_adj == state.tour_adj)
            assert np.all(orig_tour_nodes == state.tour_nodes)
        else:
            assert is_valid_tour(state.tour_nodes, state.num_nodes)
            # the swapped edges no longer exist
            assert np.all(state.tour_adj[e_swap[0], e_swap[1]] == 0)
            assert np.all(state.tour_adj[e_swap[1], e_swap[0]] == 0)
            # if orig_tour_adj[i, j] -> state.tour_adj[i, j]
            # apart from the removed edges
            assert np.all(
                np.logical_or(
                    # either the edges have been swapped out
                    swapped_tour_adj_mask,
                    # or the tour is the same
                    np.logical_or(np.logical_not(orig_tour_adj), state.tour_adj)
                )
            )
