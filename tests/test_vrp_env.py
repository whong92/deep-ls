import numpy as np
from scipy.spatial.distance import pdist, squareform
from deepls.VRPState import (
    VRPState,
    enumerate_2_opt_neighborhood,
    enumerate_all_tours_edges,
    enumerate_cross_heuristic_neighborhood,
    enumerate_relocate_neighborhood
)


def _make_random_single_tour(N = 50):
    nodes = np.random.randn(N + 1, 2)
    rand_tour = np.random.permutation(N) + 1
    edge_weights = squareform(pdist(nodes, metric='euclidean'))
    return nodes, rand_tour, edge_weights


def _make_double_loop_tour(N = 10):
    nodes = np.random.randn(N * 2 + 1, 2)
    nodes_idxs = np.random.permutation(N * 2) + 1
    split = np.random.randint(1, N * 2)
    rand_tour0 = nodes_idxs[:split]
    rand_tour1 = nodes_idxs[split:]
    demands = np.ones(shape=(N * 2))
    return nodes, rand_tour0, rand_tour1, demands


def test_double_tour_reloc():
    N = 10
    nodes, rand_tour0, rand_tour1, demands = _make_double_loop_tour(N)
    def orig_state():
        return VRPState(nodes, node_demands=demands, tours_init=[rand_tour0, rand_tour1])

    state = orig_state()
    tour_edges, _ = enumerate_all_tours_edges(
        state.tour_idx_to_tour(), directed=True
    )

    nbhs = enumerate_relocate_neighborhood(state.tour_idx_to_tour(), tour_edges)
    for nbh in nbhs:
        tour0 = nbh['src_tour']
        tour1 = nbh['dst_tour']
        node_edge_pairs = nbh['node_edges_pairs']
        for node_edge_pair in node_edge_pairs:
            src_node = node_edge_pair[0]
            src_u = node_edge_pair[1:3]
            src_v = node_edge_pair[3:5]
            dst_w = node_edge_pair[5:7]

            state._apply_relocate_move(
                tour0, tour1, src_node, src_u, src_v, dst_w
            )
            # set -1's to zero
            src_u[src_u == -1] = 0
            src_v[src_v == -1] = 0
            dst_w[dst_w == -1] = 0
            new_adj = state.get_tours_adj(directed=False)
            # relocate from single tour to depot
            # if left endpoint does not match dst edge endpoints (meaning depots)
            if src_u[0] != dst_w[0] and src_u[0] != dst_w[1]:
                assert new_adj[src_u[0], src_node] == 0
            else:
                assert new_adj[src_u[0], src_node] == 1

            # if right endpoint does not match dst edge endpoints (meaning depots)
            if src_v[1] != dst_w[0] and src_v[1] != dst_w[1]:
                assert new_adj[src_node, src_v[1]] == 0
            else:
                assert new_adj[src_node, src_v[1]] == 1

            assert new_adj[dst_w[0], src_node] == 1
            assert new_adj[src_node, dst_w[1]] == 1

            state = orig_state()


def test_double_tour_state_cross():
    N = 10
    nodes, rand_tour0, rand_tour1, demands = _make_double_loop_tour(N)

    # TODO: iterate over different N, and different splits, maybe separate out edge cases e.g. single tours / noops,
    #  maybe even remove noops
    def orig_state():
        return VRPState(nodes, node_demands=demands, tours_init=[rand_tour0, rand_tour1])

    state = orig_state()
    tour_edges, _ = enumerate_all_tours_edges(
        state.tour_idx_to_tour(), directed=True
    )

    orig_adj = state.get_tours_adj()
    # returns a list
    nbhs = enumerate_cross_heuristic_neighborhood(tour_edges)
    for nbh in nbhs:
        tour0 = nbh['src_tour']
        tour1 = nbh['dst_tour']
        edge_pairs = nbh['tour_edges_pairs']
        is_tour0_single_node = state.get_tour_lens(exclude_depot=True)[tour0] == 1
        is_tour1_single_node = state.get_tour_lens(exclude_depot=True)[tour1] == 1
        for edge_pair in edge_pairs:
            # test first move
            state = orig_state()
            e0 = edge_pair[0:2].copy()
            e1 = edge_pair[2:4].copy()
            e0p = edge_pair[4:6].copy()
            e1p = edge_pair[6:8].copy()

            state._apply_cross_move(
                tour0, tour1, e0, e1, e0p, e1p
            )
            # set -1's to zero
            e0[e0 == -1] = 0
            e1[e1 == -1] = 0
            e0p[e0p == -1] = 0
            e1p[e1p == -1] = 0
            new_adj = state.get_tours_adj(directed=False)
            # if both edges point to depot, and new edges both point to depot as well
            no_op = np.sort(e0)[0] == np.sort(e1)[0] == 0 and np.sort(e0p)[0] == np.sort(e1p)[0] == 0

            # check e0, e1 absent, except in the special case of single node tours,
            # since they have repeated node -> depot edges, and no-ops
            if not no_op:
                if not is_tour0_single_node:
                    assert new_adj[e0[0], e0[1]] == new_adj[e0[1], e0[0]] == 0
                if not is_tour1_single_node:
                    assert new_adj[e1[0], e1[1]] == new_adj[e1[1], e1[0]] == 0
            else:
                # if it is a no-op we check that nothing has changed
                assert new_adj[e0[0], e0[1]] == new_adj[e0[1], e0[0]] == 1
                assert new_adj[e1[0], e1[1]] == new_adj[e1[1], e1[0]] == 1

            if np.all(e0p == 0):  # no self loops into depot
                assert new_adj[e0p[0], e0p[1]] == new_adj[e0p[1], e0p[0]] == 0
            else:
                assert new_adj[e0p[0], e0p[1]] == new_adj[e0p[1], e0p[0]] == 1
            if np.all(e1p == 0):  # no self loops into depot
                assert new_adj[e1p[0], e1p[1]] == new_adj[e1p[1], e1p[0]] == 0
            else:
                assert new_adj[e1p[0], e1p[1]] == new_adj[e1p[1], e1p[0]] == 1

            # test 2nd move
            state = orig_state()
            e0 = edge_pair[0:2].copy()
            e1 = edge_pair[2:4].copy()
            e0p = edge_pair[8:10].copy()
            e1p = edge_pair[10:12].copy()
            state._apply_cross_move(
                tour0, tour1, e0, e1, e0p, e1p
            )
            # set -1's to zero
            e0[e0 == -1] = 0
            e1[e1 == -1] = 0
            e0p[e0p == -1] = 0
            e1p[e1p == -1] = 0
            new_adj = state.get_tours_adj(directed=False)
            # if both edges point to depot, and new edges both point to depot as well
            no_op = np.sort(e0)[0] == np.sort(e1)[0] == 0 and np.sort(e0p)[0] == np.sort(e1p)[0] == 0

            # check e0, e1 absent, except in the special case of single node tours,
            # since they have repeated node -> depot edges, and no-ops
            if not no_op:
                if not is_tour0_single_node:
                    assert new_adj[e0[0], e0[1]] == new_adj[e0[1], e0[0]] == 0
                if not is_tour1_single_node:
                    assert new_adj[e1[0], e1[1]] == new_adj[e1[1], e1[0]] == 0
            elif not no_op:
                # check e0, e1 absent
                assert new_adj[e0[0], e0[1]] == new_adj[e0[1], e0[0]] == 0
                assert new_adj[e1[0], e1[1]] == new_adj[e1[1], e1[0]] == 0
            else:
                assert new_adj[e0[0], e0[1]] == new_adj[e0[1], e0[0]] == 1
                assert new_adj[e1[0], e1[1]] == new_adj[e1[1], e1[0]] == 1

            if np.all(e0p == 0):  # no self loops into depot
                assert new_adj[e0p[0], e0p[1]] == new_adj[e0p[1], e0p[0]] == 0
            else:
                assert new_adj[e0p[0], e0p[1]] == new_adj[e0p[1], e0p[0]] == 1
            if np.all(e1p == 0):  # no self loops into depot
                assert new_adj[e1p[0], e1p[1]] == new_adj[e1p[1], e1p[0]] == 0
            else:
                assert new_adj[e1p[0], e1p[1]] == new_adj[e1p[1], e1p[0]] == 1


# @pytest.mark.parmetrize('N', [10, 20])
def test_single_tour_state_2_opt():
    N = 10
    nodes, rand_tour, edge_weights = _make_random_single_tour(N)
    demands = np.ones(shape=(N))
    state = VRPState(nodes, node_demands=demands, tours_init=[rand_tour])
    tour_edges, tour_adj = enumerate_all_tours_edges(
        state.tour_idx_to_tour(), directed=True
    )
    # orig_tour = state.all_tours_as_list(remove_last_depot=True)[0]
    # # to construct adj, remove the last -1, since the function assumes wraparound
    # orig_adj = tour_nodes_to_W(orig_tour, directed=False)
    orig_adj = state.get_tours_adj(directed=False)
    nbhs = enumerate_2_opt_neighborhood(tour_edges)

    for nbh in nbhs:
        tour_idx = nbh['tour_idx']
        tour_edges_pairs = nbh['tour_edges_pairs']
        for tour_edge_pair in tour_edges_pairs:
            e0 = tour_edge_pair[:2]
            e1 = tour_edge_pair[2:4]

            state._apply_2_opt_move(tour_idx, e0, e1)

            # set -1's to zero
            e0[e0 == -1] = 0
            e1[e1 == -1] = 0
            overlap = len(set(e0).intersection(set(e1))) != 0

            # to construct adj, remove the last -1, since the function assumes wraparound
            new_adj = state.get_tours_adj(directed=False)

            if overlap:
                assert np.all(new_adj == orig_adj)
            else:
                # mask to denote which edges have been swapped and will no longer exist
                swapped_tour_adj_mask = np.zeros_like(orig_adj)
                swapped_tour_adj_mask[e0[0], e0[1]] = 1
                swapped_tour_adj_mask[e1[0], e1[1]] = 1
                swapped_tour_adj_mask[e0[1], e0[0]] = 1
                swapped_tour_adj_mask[e1[1], e1[0]] = 1

                assert new_adj[e0[0], e0[1]] == 0
                assert new_adj[e1[0], e1[1]] == 0
                assert new_adj[e0[1], e0[0]] == 0
                assert new_adj[e0[1], e0[0]] == 0

                assert np.all(
                    np.logical_or(
                        # either the edges have been swapped out
                        swapped_tour_adj_mask,
                        # or the edge persists from the original tour i.e.
                        # if orig_tour_adj[i, j] -> state.tour_adj[i, j]
                        np.logical_or(np.logical_not(orig_adj), new_adj)
                    )
                )

            # restore to original state
            state = VRPState(nodes, node_demands=demands, tours_init=[rand_tour])