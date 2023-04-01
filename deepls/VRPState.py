import numpy as np
from typing import List, Optional, Dict, Tuple

import torch

import os, sys
from sklearn.metrics.pairwise import euclidean_distances
from graph_utils import tour_nodes_to_tour_len

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

def tour_nodes_to_node_rep(tour_nodes):
    # Compute node representation of tour
    nodes_target = {}
    for idx in range(len(tour_nodes) - 1):
        i = tour_nodes[idx]
        j = tour_nodes[idx + 1]
        nodes_target[i] = idx  # node targets: ordering of nodes in tour
    # Add final connection of tour in edge target
    nodes_target[j] = len(tour_nodes) - 1
    return nodes_target


def tour_nodes_to_cum_demands(tour_nodes: np.ndarray, demands: np.ndarray):
    non_depot_tour_nodes = tour_nodes[1:-1]
    # tour_nodes are 1-indexed, demands are 0-indexed (depot has 0 demand)
    tour_node_demands = demands[non_depot_tour_nodes - 1]
    cum_demands = np.cumsum(tour_node_demands)
    cum_demands_dict = {0: 0, -1: cum_demands[-1]}
    for n, d in zip(non_depot_tour_nodes, cum_demands):
        cum_demands_dict[n] = d
    return cum_demands_dict


def enumerate_all_tours_edges(
    tours: Dict[int, np.ndarray],
    directed=False
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Helper function to convert ordered list of tour nodes to edge adjacency matrix.
    """
    N = sum([len(tour) for tour in tours.values()])
    W = np.zeros((N + 1, N + 1))
    all_tours_edges = {}
    for tour_idx, tour in tours.items():
        edges = enumerate_tour_edges(tour, directed=directed)
        all_tours_edges[tour_idx] = edges
        for u, v in edges:
            W[u, v] = 1
    return all_tours_edges, W


def enumerate_tour_edges(nodes: np.ndarray, directed=False):
    N = len(nodes)
    assert N > 0

    edges = []
    # edit: assumed added
    # Add initial connection from depot
    # edges = [(0, int(nodes[0]))]
    # if not directed:
    #     edges.append((int(nodes[0]), 0))
    for idx in range(len(nodes) - 1):
        i = int(nodes[idx])
        j = int(nodes[idx + 1])
        edges.append((i, j))
        if not directed:
            edges.append((j, i))
    # edit: assumed added
    # Add final connection of tour to depot
    # edges.append((int(nodes[N - 1]), 0))
    # edges.append((0, int(nodes[N - 1])))
    return np.array(edges)


def enumerate_2_opt_neighborhood(tours_edges_directed: Dict[int, np.ndarray]):
    tours_edges_pairs = []
    for tour_idx, tour_edges in tours_edges_directed.items():
        # tour_edges shaped T x 2
        T, _ = tour_edges.shape
        tour_edges_pairs = np.concatenate([
            np.tile(tour_edges[:, None, :], (1, T, 1)),
            np.tile(tour_edges[None, :, :], (T, 1, 1))
        ], axis=2)  # T x T x 4 - last dim is u, v, x, y
        # indices of a T x T matrix where j > i so we don't take duplicate pairs
        rs, cs = np.triu_indices(T, k=1)
        # TODO: also filter out adjacent edges?
        tour_edges_pairs = tour_edges_pairs[rs, cs, :]
        tour_edges_pairs = np.concatenate(
            (
                tour_edges_pairs,
                tour_edges_pairs[:, [0, 2, 1, 3]],  # u, x, v, y
            ),
            axis=1
        )
        tours_edges_pairs.append({
            'tour_idx': tour_idx,
            'tour_edges_pairs': tour_edges_pairs # last dim is uvxy, uxvy
        })
    return tours_edges_pairs


def enumerate_cross_heuristic_neighborhood(tours_edges_directed_dict: Dict[int, np.ndarray]):
    neighborhood = []
    # tour_idxs = []
    # tours_edges_directed_list = []
    # for tour_idx, tour_edges_directed in tours_edges_directed_dict.items():
    #     tour_idxs.append(tour_idx)
    #     tours_edges_directed_list.append(tour_edges_directed)

    # for i, (src_tour, src_tour_edges) in enumerate(zip(tour_idxs, tours_edges_directed_list)):
    for src_tour, src_tour_edges in tours_edges_directed_dict.items():
        # for dst_tour, dst_tour_edges in zip(tour_idxs[i + 1:], tours_edges_directed_list[i + 1:]): # enumerate(tours_edges_directed[src_tour + 1:], src_tour + 1):
        for dst_tour, dst_tour_edges in tours_edges_directed_dict.items():
            if dst_tour <= src_tour:  # is symmetric neighborhood, so no need to consider anything below diagonal
                continue
            T_src = src_tour_edges.shape[0]
            T_dst = dst_tour_edges.shape[0]
            tour_edges_pairs = np.concatenate([
                np.tile(src_tour_edges[:, None, :], (1, T_dst, 1)),
                np.tile(dst_tour_edges[None, :, :], (T_src, 1, 1))
            ], axis=2)  # T_src x T_dst x 4 - last dim is u, v, x, y
            tour_edges_pairs = np.reshape(tour_edges_pairs, (-1, 4))
            tour_edges_pairs = np.concatenate(
                (
                    tour_edges_pairs,
                    tour_edges_pairs[:, [0, 2, 1, 3]],  # u, x, v, y
                    tour_edges_pairs[:, [0, 3, 1, 2]],  # u, y, v, x
                ),
                axis=1
            )
            neighborhood.append({
                'src_tour': src_tour,
                'dst_tour': dst_tour,
                'tour_edges_pairs': tour_edges_pairs
            })

    return neighborhood


def enumerate_relocate_neighborhood(tours: Dict[int, np.ndarray], tours_edges_directed: Dict[int, np.ndarray]):
    neighborhood = []
    # for src_tour_idx, (src_tour, src_edges) in enumerate(zip(tours, tours_edges_directed)):
    for src_tour_idx in tours.keys():
        src_tour = tours[src_tour_idx]
        src_edges = tours_edges_directed[src_tour_idx]
        # we skip the depot nodes (because you can't move those)
        src_nodes = src_tour[1:-1]
        T_src = src_nodes.shape[0]
        # add src node edges for convenience, the edges need to be in order of the tour
        src_edges_cat = np.concatenate([
            src_edges[0:-1],  # inbound edges
            src_edges[1:],  # outbound edges
        ], axis=1)
        assert np.all(src_edges_cat[:, 1] == src_nodes) and np.all(src_edges_cat[:, 2] == src_nodes)

        # relocate to depot move
        node_edges_pairs = np.concatenate([
            np.tile(src_nodes[:, None, None], (1, 1, 1)),
            np.tile(src_edges_cat[:, None, :], (1, 1, 1)),
            np.tile(np.array([[[0, -1]]]), (T_src, 1, 1))
        ], axis=2)
        node_edges_pairs = np.reshape(node_edges_pairs, (-1, 7))
        neighborhood.append({
            'src_tour': src_tour_idx,
            'dst_tour': None,
            'node_edges_pairs': node_edges_pairs
        })

        # relocate to other tour move
        # for dst_edges_idx, dst_edges in enumerate(tours_edges_directed):
        for dst_edges_idx, dst_edges in tours_edges_directed.items():
            if src_tour_idx == dst_edges_idx:
                continue
            T_dst = dst_edges.shape[0]
            node_edges_pairs = np.concatenate([
                np.tile(src_nodes[:, None, None], (1, T_dst, 1)),
                np.tile(src_edges_cat[:, None, :], (1, T_dst, 1)),
                np.tile(dst_edges[None, :, :], (T_src, 1, 1))
            ], axis=2)
            node_edges_pairs = np.reshape(node_edges_pairs, (-1, 7))

            neighborhood.append({
                'src_tour': src_tour_idx,
                'dst_tour': dst_edges_idx,
                'node_edges_pairs': node_edges_pairs
            })
    return neighborhood


def perform_swap(tour_nodes: np.array, i, k):
    # k larger than i, and k > i + 1, ie the two edges do not share a node
    N = len(tour_nodes)
    assert k - i >= 2 and N - k + i >= 2

    # len v -> x
    nvx = k - (i + 1)
    # len y -> u
    if (k + 1) % N == (k + 1):
        nyu = i + N - (k + 1)
    else:
        nyu = i - (k + 1) % N

    # smaller wraparound, rotate the tour, this creates a new array
    if nvx > nyu:
        tour_nodes = np.concatenate((tour_nodes[k:], tour_nodes[:k]))
        i_old = i
        i = 0
        k = i_old + len(tour_nodes[k:])

    if k == N - 1:
        tour_nodes = np.concatenate((tour_nodes[k:], tour_nodes[:k]))
        i_old = i
        i = 0
        k = i_old + len(tour_nodes[k:])

    # the segment u -> y
    segment = tour_nodes[i:k + 2]
    # reverse x -> v
    nodes_to_rev = segment[1:-1]
    tour_nodes[i + 1:k + 1] = nodes_to_rev[::-1]

    return tour_nodes


def apply_move_2_opt(e1: np.ndarray, e2: np.ndarray, nodes_pos, tour_nodes):
    u, v = e1
    x, y = e2

    # assume nodes are always facing forwards in the tour
    pos_u = nodes_pos[u]
    pos_v = nodes_pos[v]
    pos_x = nodes_pos[x]
    pos_y = nodes_pos[y]
    assert pos_u < pos_v and pos_x < pos_y, f"({u}, {v}, {x}, {y}), ({pos_u}, {pos_v}, {pos_x}, {pos_y})"

    if pos_x < pos_u:
        pos_u, pos_v, pos_x, pos_y = pos_x, pos_y, pos_u, pos_v

    N = len(tour_nodes)
    # following check implies both edges share a node
    if pos_v == pos_x:
        # we're done
        return tour_nodes, nodes_pos

    # perform swap
    # tour_nodes = perform_swap(tour_nodes, pos_u, pos_x)
    tour_nodes = perform_swap_2_opt_new(tour_nodes, pos_u, pos_v, pos_x, pos_y)
    nodes_pos = tour_nodes_to_node_rep(tour_nodes)
    return tour_nodes, nodes_pos


def perform_swap_2_opt_new(tour_nodes, u, v, x, y):
    """

    :param tour_nodes:
    u, v, x, y here are positions, not node labels
    :return:
    """
    assert u < v < x < y
    assert v == u+1
    assert y == x+1
    x1 = tour_nodes[:u+1]
    x2 = tour_nodes[v:x+1]
    x3 = tour_nodes[y:]
    return np.concatenate((x1, x2[::-1], x3))


from typing import Dict
def apply_cross_move(
    tour_0: np.ndarray,
    node_pos_0: Dict[int, int],
    tour_1: np.ndarray,
    node_pos_1: Dict[int, int],
    e0,
    e1,
    e0p,
    e1p
):
    u, v = e0
    x, y = e1
    assert u in node_pos_0 and v in node_pos_0
    assert x in node_pos_1 and y in node_pos_1
    assert e0p[0] == u
    assert e1p[0] == v

    posu = node_pos_0[u]
    posv = node_pos_0[v]
    posx = node_pos_1[x]
    posy = node_pos_1[y]
    # case ux vy
    if e0p[1] == x and e1p[1] == y:
        tour_0_new = np.concatenate((tour_0[:posu+1], tour_1[:posx+1][::-1]))
        tour_1_new = np.concatenate((tour_0[posv:][::-1], tour_1[posy:]))
    # case uy vx
    elif e0p[1] == y and e1p[1] == x:
        tour_0_new = np.concatenate((tour_0[:posu+1], tour_1[posy:]))
        tour_1_new = np.concatenate((tour_1[:posx+1], tour_0[posv:]))
    else:
        raise ValueError("invalid cross move")

    # sanity checks for endpoints
    assert tour_0_new[0] == 0 or tour_0_new[0] == -1
    assert tour_0_new[-1] == 0 or tour_0_new[-1] == -1
    assert tour_1_new[0] == 0 or tour_1_new[0] == -1
    assert tour_1_new[-1] == 0 or tour_1_new[-1] == -1
    # restore correct endpoints (0 for depot at start -1 for depot at end)
    tour_0_new[0] = 0
    tour_0_new[-1] = -1
    tour_1_new[0] = 0
    tour_1_new[-1] = -1

    node_pos_0 = tour_nodes_to_node_rep(tour_0_new)
    node_pos_1 = tour_nodes_to_node_rep(tour_1_new)

    return tour_0_new, tour_1_new, node_pos_0, node_pos_1

    # sanity checks - do in test instead?
    # assert set(tour_0_new).union(tour_1_new) == set(tour_0) == set(tour_1)
    # construct adj mat, check edges
    # calculate cost in both ways (from scratch / incrementally) and compare


def apply_relocate_move(
    tour_0: np.ndarray,
    node_pos_0: Dict[int, int],
    tour_1: Optional[np.ndarray],
    node_pos_1: Optional[Dict[int, int]],
    src_node: int,  # the node to relocate
    dst_edge: np.array,  # destination edge
):
    dst_u, dst_v = dst_edge
    dst_u_pos = node_pos_1[dst_u]
    dst_v_pos = node_pos_1[dst_v]
    assert dst_u_pos + 1 == dst_v_pos
    src_node_pos = node_pos_0[src_node]
    new_tour_0 = np.concatenate([
        tour_0[:src_node_pos],  # src_node exclusive
        tour_0[src_node_pos + 1:]
    ])
    if tour_1 is not None:
        new_tour_1 = np.concatenate([
            tour_1[:dst_u_pos + 1],  # dst_u inclusive
            np.array([src_node]),
            tour_1[dst_v_pos:]  # dst_v_incluseve
        ])
    else:
        new_tour_1 = np.array([0, src_node, -1])

    new_tour_0_pos = tour_nodes_to_node_rep(new_tour_0)
    new_tour_1_pos = tour_nodes_to_node_rep(new_tour_1)

    return new_tour_0, new_tour_1, new_tour_0_pos, new_tour_1_pos


class VRPState:
    def __init__(
        self,
        nodes_coord: np.ndarray,  # depot is always node 0,
        node_demands: np.ndarray,
        max_tour_demand: float,
        tours_init: Optional[List[np.ndarray]] = None,
        id = None
    ):
        self.nodes_coord = nodes_coord
        self.edge_weights = euclidean_distances(nodes_coord)
        self.node_demands = node_demands
        self.max_tour_demand = max_tour_demand
        self.N = len(self.nodes_coord) - 1
        self.id =id
        if tours_init:
            assert sum([len(tour) for tour in tours_init]) == self.N
            assert np.all(
                np.sort(np.concatenate(tours_init)) == np.arange(1, self.N + 1)
            )
            self.tours = {}
            for tour_idx, tour in enumerate(tours_init):
                tour = np.insert(tour, 0, 0)
                tour = np.insert(tour, len(tour), -1)
                self.tours[tour_idx] = {
                    'tour': tour,
                    'node_pos': tour_nodes_to_node_rep(tour),
                    'cum_dems': tour_nodes_to_cum_demands(tour, self.node_demands)
                }
        else:
            # every node will have its own tour
            self.tours = {}
            for i in range(1, self.N + 1):
                tour = np.array([0, i, -1])
                self.tours[i - 1] = {
                    'tour': tour,
                    'node_pos': tour_nodes_to_node_rep(tour),
                    'cum_dems': tour_nodes_to_cum_demands(tour, self.node_demands)
                }
        self.nbh = self.make_nbh()

    def all_tours_as_list(self, remove_last_depot=False):
        all_tours_list = [t['tour'] for t in self.tours.values()]
        if remove_last_depot:
            all_tours_list = [t[:-1] for t in all_tours_list]
        return all_tours_list

    def tour_idx_to_tour(self, remove_last_depot=False) -> Dict[int, np.ndarray]:
        all_tours_dict = {ti: t['tour'] for ti, t in self.tours.items()}
        if remove_last_depot:
            all_tours_dict = {ti: t[:-1] for ti, t in all_tours_dict.items()}
        return all_tours_dict

    def get_tours_adj(self, directed=False, sum=False):
        tours = self.all_tours_as_list(remove_last_depot=True)
        W = np.zeros((self.N + 1, self.N + 1))
        for tour in tours:
            for idx in range(len(tour) - 1):
                i = int(tour[idx])
                j = int(tour[idx + 1])
                if sum:
                    W[i][j] += 1
                else:
                    W[i][j] = 1
                if not directed:
                    if sum:
                        W[j][i] += 1
                    else:
                        W[j][i] = 1
                # Add final connection of tour in edge target
            if sum:
                W[j][int(tour[0])] += 1
            else:
                W[j][int(tour[0])] = 1
            if not directed:
                if sum:
                    W[int(tour[0])][j] += 1
                else:
                    W[int(tour[0])][j] = 1
        return W

    def get_tour_lens(self, exclude_depot=True):
        all_tours_len = {ti: len(t['tour']) for ti, t in self.tours.items()}
        if exclude_depot:
            all_tours_len = {ti: t - 2 for ti, t in all_tours_len.items()}
        return all_tours_len

    def get_cost(self, exclude_depot=True):
        w = self.edge_weights
        cost = 0.
        for ti, t in self.tours.items():
            tour_normalized = t['tour'].copy()
            tour_normalized[tour_normalized == -1] = 0
            cost += tour_nodes_to_tour_len(tour_normalized, w)
        return cost

    def apply_move(
        self, nb: Dict
    ):
        nb_type = nb['nb_type']
        if nb_type == 'cross':
            self._apply_cross_move(
                nb['tour0'],
                nb['tour1'],
                nb['e0'],
                nb['e1'],
                nb['e0p'],
                nb['e1p']
            )
        elif nb_type == 'reloc':
            self._apply_relocate_move(
                nb['tour0'],
                nb['tour1'],
                nb['src_node'],
                nb['src_u'],
                nb['src_v'],
                nb['dst_w']
            )
        elif nb_type == '2opt':
            self._apply_2_opt_move(
                nb['tour_idx'],
                nb['e0'],
                nb['e1']
            )
        else:
            raise ValueError("Invalid move type")
        self.nbh = self.make_nbh()

    def _apply_cross_move(
        self,
        tour_0_idx,
        tour_1_idx,
        e0, e1, e0p, e1p
    ):
        tour_0 = self.tours.pop(tour_0_idx)
        tour_1 = self.tours.pop(tour_1_idx)

        new_tour_0, new_tour_1, new_tour_0_pos, new_tour_1_pos = \
            apply_cross_move(
                tour_0['tour'], tour_0['node_pos'],
                tour_1['tour'], tour_1['node_pos'],
                e0, e1, e0p, e1p
            )
        if len(new_tour_0) > 2:
            self.tours[tour_0_idx] = {
                'tour': new_tour_0,
                'node_pos': new_tour_0_pos,
                'cum_dems': tour_nodes_to_cum_demands(new_tour_0, self.node_demands)
            }
        if len(new_tour_1) > 2:
            self.tours[tour_1_idx] = {
                'tour': new_tour_1,
                'node_pos': new_tour_1_pos,
                'cum_dems': tour_nodes_to_cum_demands(new_tour_1, self.node_demands)
            }

    def _apply_2_opt_move(
        self,
        tour_idx, e0, e1
    ):
        tour = self.tours.pop(tour_idx)
        new_tour_nodes, new_node_pos = apply_move_2_opt(
            e0, e1,
            tour['node_pos'], tour['tour'],
        )
        self.tours[tour_idx] = {
            'tour': new_tour_nodes,
            'node_pos': new_node_pos,
            'cum_dems': tour_nodes_to_cum_demands(new_tour_nodes, self.node_demands)
        }

    def _apply_relocate_move(
        self,
        src_tour_idx,
        dst_tour_idx,
        src_node,
        src_u,
        src_v,
        dst_w
    ):
        # move relocates to empty tour, create new dst_tour
        if dst_tour_idx is None:
            dst_tour_idx = max(self.tours.keys()) + 1
            dst_tour = {
                'tour': np.array([0, -1]),
                'node_pos': {
                    0: 0, -1: 1
                }
            }
        else:
            dst_tour = self.tours.pop(dst_tour_idx)

        src_tour = self.tours.pop(src_tour_idx)
        dst_tour_nodes = dst_tour['tour']
        dst_tour_node_pos = dst_tour['node_pos']
        src_tour_nodes = src_tour['tour']
        src_tour_node_pos = src_tour['node_pos']

        src_tour_nodes_new, dst_tour_nodes_new, src_tour_nodes_pos_new, dst_tour_nodes_pos_new \
            = apply_relocate_move(
                src_tour_nodes,
                src_tour_node_pos,
                dst_tour_nodes,
                dst_tour_node_pos,
                src_node,
                dst_w
            )

        if len(src_tour_nodes_new) > 2:
            self.tours[src_tour_idx] = {
                'tour': src_tour_nodes_new,
                'node_pos': src_tour_nodes_pos_new,
                'cum_dems': tour_nodes_to_cum_demands(src_tour_nodes_new, self.node_demands)
            }
        else:
            assert np.all(src_tour_nodes_new == np.array([0, -1])), "just double checking"

        self.tours[dst_tour_idx] = {
            'tour': dst_tour_nodes_new,
            'node_pos': dst_tour_nodes_pos_new,
            'cum_dems': tour_nodes_to_cum_demands(dst_tour_nodes_new, self.node_demands)
        }

    def make_nbh(self):
        return VRPNbH(self)

    def get_nbh(self):
        return self.nbh


def normalize_edge(e, as_tuple=True):
    e = e.copy()
    # normalize it by changing -1's to 0's
    e[e == -1] = 0
    # and sorting
    e = np.sort(e)
    if as_tuple:
        # make them tuples to be hashable
        return tuple(e)
    return e


def get_normalized_nbh_rep(src_edges: List[np.ndarray], dst_edges: List[np.ndarray]):
    src_edges = filter(lambda e: e[0] != e[1], [normalize_edge(e) for e in src_edges])
    dst_edges = filter(lambda e: e[0] != e[1], [normalize_edge(e) for e in dst_edges])
    src_edges = set(src_edges)
    dst_edges = set(dst_edges)
    edges_added = list(src_edges.difference(dst_edges))
    edges_removed = list(dst_edges.difference(src_edges))
    edges_added = tuple(sorted(edges_added, key=lambda e: e[0]))
    edges_removed = tuple(sorted(edges_removed, key=lambda e: e[0]))
    edge_diff = (edges_added, edges_removed)
    return edge_diff


def check_no_op(src_edges: List[np.ndarray], dst_edges: List[np.ndarray]):
    edge_diff = get_normalized_nbh_rep(src_edges, dst_edges)
    return len(edge_diff) == 0


def check_cross_move_valid(
    e0,
    e1,
    e0p,
    e1p,
    cum_demand_0,
    cum_demand_1,
    max_tour_demand
):
    u, v = e0
    x, y = e1
    assert u in cum_demand_0 and v in cum_demand_0
    assert x in cum_demand_1 and y in cum_demand_1
    assert e0p[0] == u
    assert e1p[0] == v

    cum_dem_u = cum_demand_0[u]
    cum_dem_v = cum_demand_0[v]
    cum_dem_x = cum_demand_1[x]
    cum_dem_y = cum_demand_1[y]

    C0 = cum_demand_0[-1]  # total demand for tour 0
    C1 = cum_demand_1[-1]  # total demand for tour 1

    assert cum_dem_u <= cum_dem_v and cum_dem_x <= cum_dem_y
    # case ux vy
    if e0p[1] == x and e1p[1] == y:
        # 0 -> u -> x -> 0
        tour_0_new_dem = cum_dem_u + cum_dem_x
        # -1 -> v -> y -> -1
        tour_1_new_dem = (C0 - cum_dem_u) + (C1 - cum_dem_x)
    # case uy vx
    elif e0p[1] == y and e1p[1] == x:
        # 0 -> u -> y -> -1
        tour_0_new_dem = cum_dem_u + (C1 - cum_dem_x)
        # 0 -> x -> v -> -1
        tour_1_new_dem = cum_dem_x + (C0 - cum_dem_u)
    else:
        raise ValueError("invalid cross move")

    assert tour_1_new_dem + tour_0_new_dem == C0 + C1
    return (tour_0_new_dem <= max_tour_demand) and (tour_1_new_dem <= max_tour_demand)


def check_relocate_move_valid(
    src_node,
    cum_demand_dst,
    demands,
    max_tour_dem
):
    dst_demand = cum_demand_dst[-1]
    src_demand = demands[src_node - 1]
    return (dst_demand + src_demand) <= max_tour_dem


class VRPNbH:

    @staticmethod
    def enumerate_all_nbs(state: VRPState):
        # relocation heuristic
        tour_edges, _ = enumerate_all_tours_edges(
            state.tour_idx_to_tour(), directed=True
        )
        nbhs = enumerate_relocate_neighborhood(state.tour_idx_to_tour(), tour_edges)
        relocate_nbhs = {}
        for nbh in nbhs:
            tour0 = nbh['src_tour']
            tour1 = nbh['dst_tour']
            node_edge_pairs = nbh['node_edges_pairs']
            for node_edge_pair in node_edge_pairs:
                src_node = node_edge_pair[0]
                src_u = node_edge_pair[1:3]
                src_v = node_edge_pair[3:5]
                dst_w = node_edge_pair[5:7]
                src_edges = [src_u, src_v, dst_w]
                dst_edges = [
                    np.array([src_u[0], src_v[1]]),
                    np.array([dst_w[0], src_node]),
                    np.array([src_node, dst_w[1]]),
                ]
                nbh_norm = get_normalized_nbh_rep(src_edges, dst_edges)
                no_op = len(nbh_norm) == 0

                if (tour1 is not None) and (not check_relocate_move_valid(
                        src_node,
                        state.tours[tour1]['cum_dems'],
                        state.node_demands,
                        max_tour_dem=state.max_tour_demand
                )):
                    continue

                # remove no_ops and duplicates
                if not no_op and (nbh_norm not in relocate_nbhs):
                    nb = {
                        "nb_type": "reloc",
                        "tour0": tour0,
                        "tour1": tour1,
                        "src_node": src_node,
                        "src_u": src_u,
                        "src_v": src_v,
                        "dst_w": dst_w,
                        "dst_wp": dst_edges[0],
                        "src_up": dst_edges[1],
                        "src_vp": dst_edges[2]
                    }
                    relocate_nbhs[nbh_norm] = nb

        # cross heuristic
        nbhs = enumerate_cross_heuristic_neighborhood(tour_edges)
        cross_h_nbhs = {}
        for nbh in nbhs:
            tour0 = nbh['src_tour']
            tour1 = nbh['dst_tour']
            edge_pairs = nbh['tour_edges_pairs']

            cum_dem_0 = state.tours[tour0]['cum_dems']
            cum_dem_1 = state.tours[tour1]['cum_dems']

            for edge_pair in edge_pairs:
                # first move
                e0 = edge_pair[0:2]
                e1 = edge_pair[2:4]
                e0p = edge_pair[4:6]
                e1p = edge_pair[6:8]
                nbh_norm = get_normalized_nbh_rep([e0, e1], [e0p, e1p])

                # print(cum_dem_0, e0)
                # print(cum_dem_1, e1)
                no_op = len(nbh_norm) == 0
                # remove no_ops and duplicates
                if not no_op and (nbh_norm not in cross_h_nbhs) and check_cross_move_valid(e0, e1, e0p, e1p, cum_dem_0,
                                                                                           cum_dem_1,
                                                                                           state.max_tour_demand):
                    nb = {
                        "nb_type": "cross",
                        "tour0": tour0,
                        "tour1": tour1,
                        "e0": e0,
                        "e1": e1,
                        "e0p": e0p,
                        "e1p": e1p,
                    }
                    cross_h_nbhs[nbh_norm] = nb
                # second move
                e0p = edge_pair[8:10]
                e1p = edge_pair[10:12]
                nbh_norm = get_normalized_nbh_rep([e0, e1], [e0p, e1p])

                # print(cum_dem_0, e0)
                # print(cum_dem_1, e1)
                no_op = len(nbh_norm) == 0
                # remove no_ops and duplicates
                if not no_op and (nbh_norm not in cross_h_nbhs) and check_cross_move_valid(e0, e1, e0p, e1p, cum_dem_0,
                                                                                           cum_dem_1,
                                                                                           state.max_tour_demand):
                    nb = {
                        "nb_type": "cross",
                        "tour0": tour0,
                        "tour1": tour1,
                        "e0": e0,
                        "e1": e1,
                        "e0p": e0p,
                        "e1p": e1p,
                    }
                    cross_h_nbhs[nbh_norm] = nb

        # 2opt heuristics
        nbhs = enumerate_2_opt_neighborhood(tour_edges)
        two_opt_nbhs = {}
        for nbh in nbhs:
            tour_idx = nbh['tour_idx']
            tour_edges_pairs = nbh['tour_edges_pairs']
            for tour_edge_pair in tour_edges_pairs:
                e0 = tour_edge_pair[:2]
                e1 = tour_edge_pair[2:4]
                e0p = tour_edge_pair[4:6]
                e1p = tour_edge_pair[6:8]
                nbh_norm = get_normalized_nbh_rep([e0, e1], [e0p, e1p])
                no_op = len(nbh_norm) == 0
                # remove no_ops and duplicates
                if not no_op and (nbh_norm not in cross_h_nbhs):
                    nb = {
                        "nb_type": "2opt",
                        "tour_idx": tour_idx,
                        "e0": tour_edge_pair[:2],
                        "e1": tour_edge_pair[2:4],
                        "e0p": e0p,
                        "e1p": e1p
                    }
                    two_opt_nbhs[nbh_norm] = nb

        # figure out how to make repeatable sequence generation
        nbh_list = []
        relocate_nbhs = list(relocate_nbhs.values())
        cross_h_nbhs = list(cross_h_nbhs.values())
        two_opt_nbhs = list(two_opt_nbhs.values())
        nbh_list.extend(relocate_nbhs)
        nbh_list.extend(cross_h_nbhs)
        nbh_list.extend(two_opt_nbhs)
        return nbh_list, relocate_nbhs, cross_h_nbhs, two_opt_nbhs

    def __init__(self, state: VRPState):
        nbh_list, relocate_nbhs, cross_nbhs, two_opt_nbhs = VRPNbH.enumerate_all_nbs(state)
        self.nbh_list = nbh_list
        self.relocate_nbhs_vectorized = vectorize_reloc_moves(relocate_nbhs)
        self.cross_nbhs_vectorized = vectorize_cross_moves(cross_nbhs)
        self.two_opt_nbhs_vectorized = vectorize_twopt_moves(two_opt_nbhs)

    def get_nb(self, i):
        return self.nbh_list[i]


def get_edge_embs(e_emb: torch.Tensor, edges: torch.Tensor, symmetric=True) -> torch.Tensor:
    """
    :param e_emb: b x v x v x h - edge embeddings
    :param edges: ... x 3 - (batch_idx, u, v)
    :return: ... x h
    """
    bs = edges[..., 0]
    us = edges[..., 1]
    vs = edges[..., 2]
    ret = e_emb[bs, us, vs]
    if symmetric:
        ret += e_emb[bs, vs, us]
    return ret


def vectorize_cross_moves(cross_moves):
    """
    To be cached in buffer
    :param cross_moves:
    :return: num_moves x 2
    """
    vectorized = {}
    for key in ['e0', 'e1', 'e0p', 'e1p']:
        if len(cross_moves) == 0:
            vectorized[key] = np.empty(shape=(0, 2))
        else:
            vectorized[key] = np.stack([
                normalize_edge(cross_move[key])
                for cross_move in cross_moves
            ], axis=0)
    return vectorized


def vectorize_reloc_moves(reloc_moves):
    """
    To be cached in buffer
    :param reloc_moves:
    :return: num_moves x 2
    """
    vectorized = {}
    for key in ['src_u', 'src_v', 'dst_w', 'src_up', 'src_vp', 'dst_wp']:
        if len(reloc_moves) == 0:
            vectorized[key] = np.empty(shape=(0, 2))
        else:
            vectorized[key] = np.stack([
                normalize_edge(reloc_move[key])
                for reloc_move in reloc_moves
            ], axis=0)
    return vectorized


def vectorize_twopt_moves(twopt_moves):
    """
    To be cached in buffer
    :param reloc_moves:
    :return: num_moves x 2
    """
    vectorized = {}
    for key in ['e0', 'e1', 'e0p', 'e1p']:
        if len(twopt_moves) == 0:
            vectorized[key] = np.empty(shape=(0, 2))
        else:
            vectorized[key] = np.stack([
                normalize_edge(twopt_move[key])
                for twopt_move in twopt_moves
            ], axis=0)
    return vectorized


def embed_cross_heuristic(cross_moves_vectorized, edge_embeddings: torch.Tensor, cross_move_mlp: torch.nn.Module):
    """
    list of list of cross moves
    for each state, flatten into a single e0, e1, e0p, e1p
    embed each e0, e1, e0p, e1p by getting the edge embeddings,
    and then using a cross-move MLP (each heuristic will have its own move)

    just cat the tensors
    then we embed the catted sequence
    then split again into a list of tensors

    embed relocate / 2opt moves

    :return:
    """
    e0 = []
    e1 = []
    e0p = []
    e1p = []
    num_moves = []
    for i, m in enumerate(cross_moves_vectorized):
        num_moves_i = m['e0'].shape[0]
        batch_index = i * np.ones(shape=(num_moves_i, 1))
        e0.append(np.concatenate([batch_index, m['e0']], axis=1))
        e1.append(np.concatenate([batch_index, m['e1']], axis=1))
        e0p.append(np.concatenate([batch_index, m['e0p']], axis=1))
        e1p.append(np.concatenate([batch_index, m['e1p']], axis=1))
        num_moves.append(num_moves_i)

    # sum_b ( num_moves(b) )  x  3
    e0 = torch.as_tensor(np.concatenate(e0, axis=0)).long()
    e1 = torch.as_tensor(np.concatenate(e1, axis=0)).long()
    e0p = torch.as_tensor(np.concatenate(e0p, axis=0)).long()
    e1p = torch.as_tensor(np.concatenate(e1p, axis=0)).long()
    # sum_b ( num_moves(b) ) x 1

    edges = torch.stack([e0, e1, e0p, e1p], dim=0)
    edges_emb = get_edge_embs(edge_embeddings, edges, symmetric=True)
    # sum_b ( num_moves(b) )  x  3
    e0_emb, e1_emb, e0p_emb, e1p_emb = edges_emb[0], edges_emb[1], edges_emb[2], edges_emb[3]

    # MLP the embeddings into a single embedding
    # sum_b ( num_moves(b) ) x (4 * h)
    cross_move_input_emb = torch.cat([e0_emb, e1_emb, e0p_emb, e1p_emb], dim=1)
    # MLP input dim is 4 x h output dim is h
    # sum_b ( num_moves(b) ) x (4 * h)
    cross_move_embeddings = cross_move_mlp(cross_move_input_emb)
    # split it into a list of tensors using num_moves
    assert cross_move_embeddings.shape[0] == sum(num_moves)
    cross_move_embeddings = torch.split(cross_move_embeddings, num_moves, dim=0)
    # cat it with 2opt and reloc
    return cross_move_embeddings


def embed_reloc_heuristic(reloc_moves_vectorized, edge_embeddings: torch.Tensor, reloc_move_mlp: torch.nn.Module):
    """
    list of list of cross moves
    for each state, flatten into a single e0, e1, e0p, e1p
    embed each e0, e1, e0p, e1p by getting the edge embeddings,
    and then using a cross-move MLP (each heuristic will have its own move)

    just cat the tensors
    then we embed the catted sequence
    then split again into a list of tensors

    embed relocate / 2opt moves

    :return:
    """
    src_u = []
    src_v = []
    dst_w = []
    src_up = []
    src_vp = []
    dst_wp = []
    num_moves = []
    for i, m in enumerate(reloc_moves_vectorized):
        num_moves_i = m['src_u'].shape[0]
        batch_index = i * np.ones(shape=(num_moves_i, 1))
        src_u.append(np.concatenate([batch_index, m['src_u']], axis=1))
        src_v.append(np.concatenate([batch_index, m['src_v']], axis=1))
        dst_w.append(np.concatenate([batch_index, m['dst_w']], axis=1))
        src_up.append(np.concatenate([batch_index, m['src_up']], axis=1))
        src_vp.append(np.concatenate([batch_index, m['src_vp']], axis=1))
        dst_wp.append(np.concatenate([batch_index, m['dst_wp']], axis=1))
        num_moves.append(num_moves_i)

    # sum_b ( num_moves(b) )  x  3
    src_u = torch.as_tensor(np.concatenate(src_u, axis=0)).long()
    src_v = torch.as_tensor(np.concatenate(src_v, axis=0)).long()
    dst_w = torch.as_tensor(np.concatenate(dst_w, axis=0)).long()
    src_up = torch.as_tensor(np.concatenate(src_up, axis=0)).long()
    src_vp = torch.as_tensor(np.concatenate(src_vp, axis=0)).long()
    dst_wp = torch.as_tensor(np.concatenate(dst_wp, axis=0)).long()
    # sum_b ( num_moves(b) ) x 1

    edges = torch.stack([
        src_u,
        src_v,
        dst_w,
        src_up,
        src_vp,
        dst_wp
    ], dim=0)
    # 6 x sum_b ( num_moves(b) )  x  3
    edges_emb = get_edge_embs(edge_embeddings, edges, symmetric=True)

    # there may be a more elegant way of unwrapping this but lazy
    # sum_b ( num_moves(b) )  x  3
    src_u_emb,src_v_emb, dst_w_emb, src_up_emb, src_vp_emb, dst_wp_emb = \
        edges_emb[0], edges_emb[1], edges_emb[2], edges_emb[3], edges_emb[4], edges_emb[5]

    # MLP the embeddings into a single embedding
    # sum_b ( num_moves(b) ) x (6 * h)
    reloc_move_input_emb = torch.cat([src_u_emb, src_v_emb, dst_w_emb, src_up_emb, src_vp_emb, dst_wp_emb], dim=1)
    # MLP input dim is 4 x h output dim is h
    # sum_b ( num_moves(b) ) x (4 * h)
    reloc_move_embeddings = reloc_move_mlp(reloc_move_input_emb)
    # split it into a list of tensors using num_moves
    assert reloc_move_embeddings.shape[0] == sum(num_moves)
    reloc_move_embeddings = torch.split(reloc_move_embeddings, num_moves, dim=0)
    # cat it with 2opt and reloc
    return reloc_move_embeddings


from matplotlib import pyplot as plt

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py

def discrete_cmap(N, base_cmap=None):
  """
    Create an N-bin discrete colormap from the specified input map
    """
  # Note that if base_cmap is a string or None, you can simply do
  #    return plt.cm.get_cmap(base_cmap, N)
  # The following works for string, None, or a colormap instance:

  base = plt.cm.get_cmap(base_cmap)
  color_list = base(np.linspace(0, 1, N))
  cmap_name = base.name + str(N)
  return base.from_list(cmap_name, color_list, N)


def plot_vehicle_routes(data, routes, ax1, markersize=5, visualize_demands=False, demand_scale=1, round_demand=False):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """

    # route is one sequence, separating different routes with 0 (depot)
    depot = data['depot']
    locs = data['loc']
    demands = data['demand'] * demand_scale
    capacity = demand_scale  # Capacity is always 1

    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, 'sk', markersize=markersize * 4)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    legend = ax1.legend(loc='upper center')

    cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
    dem_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    for veh_number, r in enumerate(routes):
        color = cmap(len(routes) - veh_number)  # Invert to have in rainbow order

        route_demands = demands[r - 1]
        coords = locs[r - 1, :]
        xs, ys = coords.transpose()

        total_route_demand = sum(route_demands)
        assert total_route_demand <= capacity
        if not visualize_demands:
            ax1.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)

        dist = 0
        x_prev, y_prev = x_dep, y_dep

        cum_demand = 0
        for (x, y), d in zip(coords, route_demands):
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)

            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity))
            dem_rects.append(Rectangle((x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity))

            x_prev, y_prev = x, y
            cum_demand += d

        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=color,
            label='R{}, # {}, c {} / {}, d {:.2f}'.format(
                veh_number,
                len(r),
                int(total_route_demand) if round_demand else total_route_demand,
                int(capacity) if round_demand else capacity,
                dist
            )
        )

        qvs.append(qv)

    ax1.set_title('{} routes, total distance {:.2f}'.format(len(routes), total_dist))
    # ax1.legend(handles=qvs)

    pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
    pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
    pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')

    if visualize_demands:
        ax1.add_collection(pc_cap)
        ax1.add_collection(pc_used)
        ax1.add_collection(pc_dem)

if __name__=="__main__":
    # tour_edges = [
    #     np.array([
    #         [1, 2],
    #         [0, 1],
    #         [2, 3],
    #         [3, 4],
    #         [4, 5],
    #         [5, 6],
    #         [6, 7],
    #     ]),
    #     np.array([
    #         [8, 9],
    #         [9, 10],
    #         [10, 11],
    #         [11, 12],
    #         [12, 13],
    #         [13, 14],
    #         [14, 15],
    #     ])
    # ]
    # print(enumerate_2_opt_neighborhood(tour_edges))
    #
    # print(enumerate_cross_heuristic_neighborhood(tour_edges))
    #
    # tours = [
    #     np.array([0, 1, 2, 3, 4, 5, 6, 7, -1]),
    #     np.array([0, 8, 9, 10, 11, 12, 13, 14, -1])
    # ]
    # tour_edges, tour_adj = enumerate_all_tours_edges(tours, directed=True)
    # print(tour_edges)
    # print(enumerate_cross_heuristic_neighborhood(tour_edges))
    # twoopt_neighborhood = enumerate_2_opt_neighborhood(tour_edges)
    # cross_neighborhood = enumerate_cross_heuristic_neighborhood(tour_edges)
    #
    # tours_node_pos = [tour_nodes_to_node_rep(tour) for tour in tours]
    #
    # print(" test 2 opt neighrborhood")
    # neighbor = twoopt_neighborhood[1][15]
    # e0 = neighbor[:2]
    # e1 = neighbor[2:4]
    # print(e0, e1)
    # print(tours_node_pos[1])
    # print(tours[1])
    # new_tour, new_node_pos = apply_move_2_opt(e0, e1, tours_node_pos[1], tours[1])
    # print(new_tour)
    # print(new_node_pos)
    #
    # print("test cross neighborhood")
    #
    # tour_0 = tours[cross_neighborhood[0]['src_tour']]
    # tour_1 = tours[cross_neighborhood[0]['dst_tour']]
    # tours_node_pos_0 = tour_nodes_to_node_rep(tour_0)
    # tours_node_pos_1 = tour_nodes_to_node_rep(tour_1)
    # print(tours_node_pos_0, tours_node_pos_1)
    # edge_pair = cross_neighborhood[0]['tour_edges_pairs'][14]
    # e0 = edge_pair[0:2]
    # e1 = edge_pair[2:4]
    # e0p = edge_pair[4:6]
    # e1p = edge_pair[6:8]
    # print(edge_pair)
    #
    # tour_0_new, tour_1_new, node_pos_0, node_pos_1 = apply_cross_move(
    #     tours[0],
    #     tours_node_pos_0,
    #     tours[1],
    #     tours_node_pos_1,
    #     e0,
    #     e1,
    #     e0p,
    #     e1p
    # )
    #
    # print(tour_0_new)
    # print(tour_1_new)
    #
    # coords = np.random.random(size=(len(tours[0]) + len(tours[1]), 2))
    # demands = np.ones(shape=(len(tours[0]) + len(tours[1])))
    # demands /= np.sum(demands)
    # print(coords, demands)
    #
    # tours_edges, adj = enumerate_all_tours_edges(tours, directed=True)
    # rnbh = enumerate_relocate_neighborhood(tours, tours_edges)
    # print(rnbh)
    #
    # data = {
    #     'loc': coords,
    #     'demand': demands,
    #     'depot': np.array([0.5, 0.5]),
    # }
    # fig, ax = plt.subplots(figsize=(8, 8))
    # fig2, ax2 = plt.subplots(figsize=(8, 8))
    # plot_vehicle_routes(data, tours, ax1=ax)
    #
    # plot_vehicle_routes(data, [tour_0_new, tour_1_new], ax1=ax2)
    #
    # plt.show()

    N = 10
    demands = np.ones(shape=(N))
    demands_norm = demands / np.sum(demands)
    coords = np.zeros(shape=(N + 1, 2))
    coords[0] = 0.5
    coords[1:] = np.random.random(size=(N, 2))

    state = VRPState(coords, node_demands=demands, max_tour_demand=4)

    data = {'loc': coords[1:], 'depot': coords[0], 'demand': demands_norm}

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_vehicle_routes(data, [t[1:] for t in state.all_tours_as_list(remove_last_depot=True)], ax1=ax)

    plt.show()
    from deepls.gcn_model import ResidualGatedGCNModel

    hidden_dim = 12
    config = {
        "node_dim": 2,
        "voc_edges_in": 3,
        "hidden_dim": hidden_dim,
        "num_layers": 2,
        "mlp_layers": 3,
        "aggregation": "mean",
        "num_edge_cat_features": 2
    }
    rgcn = ResidualGatedGCNModel(config)
    cross_mlp = torch.nn.Linear(in_features=4 * hidden_dim, out_features=hidden_dim)
    reloc_mlp = torch.nn.Linear(in_features=6 * hidden_dim, out_features=hidden_dim)

    import random

    for _ in range(30):
        nbh = state.make_nbh()

        random_nb = random.choice(nbh.nbh_list)
        state.apply_move(random_nb)

        adj = torch.as_tensor(
            state.get_tours_adj(sum=True).astype(int)
        ).long()[None, :, :]
        x_emb, e_emb = rgcn(
            torch.stack([adj, adj], dim=3),
            torch.as_tensor(state.edge_weights[None, :, :]).float(),
            torch.as_tensor(coords[None, :, :]).float()
        )
        print(x_emb.shape, e_emb.shape)

        reloc_moves_embedded = embed_reloc_heuristic(
            reloc_moves_vectorized=[nbh.relocate_nbhs_vectorized],
            edge_embeddings=e_emb,
            reloc_move_mlp=reloc_mlp
        )
        cross_moves_embedded = embed_cross_heuristic(
            cross_moves_vectorized=[nbh.cross_nbhs_vectorized],
            edge_embeddings=e_emb,
            cross_move_mlp=cross_mlp
        )
        two_opt_moves_embedded = embed_cross_heuristic(
            cross_moves_vectorized=[nbh.two_opt_nbhs_vectorized],
            edge_embeddings=e_emb,
            cross_move_mlp=cross_mlp
        )
        all_moves_embedded = torch.stack([
            torch.cat([r, c, t], dim=0)
            for r, c, t in
            zip(
                reloc_moves_embedded,
                cross_moves_embedded,
                two_opt_moves_embedded
            )
        ], dim=0)
        print(cross_moves_embedded[0].shape)
        print(reloc_moves_embedded[0].shape)
        print(two_opt_moves_embedded[0].shape)
        print(all_moves_embedded.shape)
        print(len(nbh.nbh_list))
        # print(reloc_moves_embedded)

        fig, ax = plt.subplots(figsize=(4, 4))
        plot_vehicle_routes(data, [t[1:] for t in state.all_tours_as_list(remove_last_depot=True)], ax1=ax)
        plt.show()


