import numpy as np
from typing import List, Optional, Dict, Tuple, Any, Union
import random
import vrpstate

from datetime import datetime
import torch

import os, sys
from sklearn.metrics.pairwise import euclidean_distances
from deepls.graph_utils import tour_nodes_to_tour_len
from enum import Enum

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


def enumerate_all_tours_nodes(tours: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    all_tours_nodes = {}
    for tour_idx, tour in tours.items():
        all_tours_nodes[tour_idx] = tour
    return all_tours_nodes


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


def enumerate_relocate_neighborhood_given(
    src_node: int,
    src_tour_idx: int,
    src_node_pos: int,
    tours_edges_directed: Dict[int, np.ndarray],
    # node_pos: Dict[int, int]
):
    neighborhood = []
    tour_edges = tours_edges_directed[src_tour_idx]
    # src_node_pos = src_node_pos[src_node]
    src_node = np.array([src_node])  # 1,

    in_edge = tour_edges[src_node_pos - 1]
    out_edge = tour_edges[src_node_pos]
    src_edges = np.concatenate([in_edge, out_edge], axis=0)  # 4
    dst_edge = np.array([[0, -1]])  # dst_w
    # relocate to depot move
    node_edges_pairs = np.concatenate([
        src_node[None, :],  # 1, 1
        src_edges[None, :],  # 1, 4
        dst_edge  # 1, 2
    ], axis=1)  # 1, 7
    neighborhood.append({
        'src_tour': src_tour_idx,
        'dst_tour': None,
        'node_edges_pairs': node_edges_pairs
    })

    # relocate to other tour move
    for dst_edges_idx, dst_edges in tours_edges_directed.items():
        if src_tour_idx == dst_edges_idx:
            continue
        T_dst = dst_edges.shape[0]
        node_edges_pairs = np.concatenate([
            np.tile(src_node[None, :], (T_dst, 1)),  # T_dst, 1
            np.tile(src_edges[None, :], (T_dst, 1)),  # T_dst, 4
            dst_edges[:, :]  # T_dst, 2
        ], axis=1)
        neighborhood.append({
            'src_tour': src_tour_idx,
            'dst_tour': dst_edges_idx,
            'node_edges_pairs': node_edges_pairs
        })

    return neighborhood


def enumerate_2_opt_neighborhood_given(
    tour_idx: int,
    src_edge: np.ndarray,  # 2
    tour_edges_directed: Dict[int, np.ndarray]
):
    tour_edges = tour_edges_directed[tour_idx]  # T, 2

    exclude = np.logical_or(
        np.any(tour_edges == src_edge[1], axis=1),
        np.any(tour_edges == src_edge[0], axis=1)
    )
    tour_edges = tour_edges[~exclude]
    if len(tour_edges) == 0:
        tour_edges_pairs = np.empty(shape=(0, 8))
    else:
        T, _ = tour_edges.shape
        src_edge = np.tile(src_edge[None, :], (T, 1))
        tour_edges_pairs = np.concatenate([
            src_edge,
            tour_edges[:, :]  # (T, 2)
        ], axis=1)  # T, 4
        # print(tour_edges_pairs)
        tour_edges_pairs = np.concatenate(
            (
                tour_edges_pairs,
                tour_edges_pairs[:, [0, 2, 1, 3]],  # u, x, v, y
            ),
            axis=1
        )
    return [{
        'tour_idx': tour_idx,
        'tour_edges_pairs': tour_edges_pairs # last dim is uvxy, uxvy
    }]


def enumerate_cross_neighborhood_given(
    src_tour: int,
    src_tour_edge: np.ndarray,  # 2
    tours_edges_directed: Dict[int, np.ndarray]
):
    neighborhood = []
    for dst_tour, dst_tour_edges in tours_edges_directed.items():
        if dst_tour == src_tour:
            continue
        T_dst = dst_tour_edges.shape[0]
        tour_edges_pairs = np.concatenate([
            np.tile(src_tour_edge[None, :], (T_dst, 1)),  # T_dst x 2
            dst_tour_edges[:, :]    # T_dst x 2
        ], axis=1)  # T_dst x 4 - last dim is u, v, x, y
        tour_edges_pairs = np.concatenate(
            (
                tour_edges_pairs,
                tour_edges_pairs[:, [0, 2, 1, 3]],  # u, x, v, y
                tour_edges_pairs[:, [0, 3, 1, 2]],  # u, y, v, x
            ),
            axis=1
        )  # T_dst x ((uvxy), (uxvy), (uyvx))
        neighborhood.append({
            'src_tour': src_tour,
            'dst_tour': dst_tour,
            'tour_edges_pairs': tour_edges_pairs
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
    assert u < v < x < y, f"{u} < {v}, {x}, {y}"
    assert v == u+1
    assert y == x+1
    x1 = tour_nodes[:u+1]
    x2 = tour_nodes[v:x+1]
    x3 = tour_nodes[y:]
    return np.concatenate((x1, x2[::-1], x3))


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


class VRPInitTour(Enum):
    SINGLETON = 'SINGETON'
    MAX_CAP_RANDOM = 'MAX_CAP_RANDOM'


class VRPState:
    # WLOG, demands should be scaled appropriately
    VEHICLE_CAPACITY = 1.0

    def get_node_demands(self, include_depot=False):
        if include_depot:
            return np.concatenate((np.array([0.]), self.node_demands))
        return self.node_demands

    @classmethod
    def copy_construct(cls, other: 'VRPState'):
        return VRPState(
            other.nodes_coord,
            other.node_demands,
            other.max_tour_demand,
            tours_init=other.all_tours_as_list(remove_last_depot=True, remove_first_depot=True),
            id=other.id,
            opt_tour_dist=other.opt_tour_dist,
            init_tour=other.init_tour
        )

    def __init__(
        self,
        nodes_coord: np.ndarray,  # depot is always node 0,
        node_demands: np.ndarray,
        max_tour_demand: Optional[float] = VEHICLE_CAPACITY,
        tours_init: Optional[List[np.ndarray]] = None,
        id=None,
        opt_tour_dist: Optional[float] = None,
        init_tour: VRPInitTour = VRPInitTour.SINGLETON
    ):
        self.nodes_coord = nodes_coord
        self.edge_weights = euclidean_distances(nodes_coord)
        self.node_demands = node_demands
        self.max_tour_demand = max_tour_demand if max_tour_demand else np.sum(node_demands)
        self.N = len(self.nodes_coord) - 1
        self.id = id
        self.opt_tour_dist = opt_tour_dist
        self.init_tour = init_tour

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
            self.tours = {}
            if self.init_tour == VRPInitTour.SINGLETON:
                # every node will have its own tour
                for i in range(1, self.N + 1):
                    tour = np.array([0, i, -1])
                    self.tours[i - 1] = {
                        'tour': tour,
                        'node_pos': tour_nodes_to_node_rep(tour),
                        'cum_dems': tour_nodes_to_cum_demands(tour, self.node_demands)
                    }
            elif self.init_tour == VRPInitTour.MAX_CAP_RANDOM:
                nodes_remaining = np.random.permutation(np.arange(1, self.N + 1))
                tours_init = []
                while len(nodes_remaining) > 0:
                    tour_init = []
                    target_demand = np.random.uniform(np.min(self.node_demands), self.max_tour_demand)
                    tour_demand = 0.
                    while len(nodes_remaining) > 0:
                        if (tour_demand + self.node_demands[nodes_remaining[0] - 1]) <= target_demand:
                            tour_init.append(nodes_remaining[0])
                            nodes_remaining = nodes_remaining[1:]
                            tour_demand += self.node_demands[tour_init[-1] - 1]
                        else:
                            break
                    tours_init.append(np.array(tour_init))
                for tour_idx, tour in enumerate(tours_init):
                    tour = np.insert(tour, 0, 0)
                    tour = np.insert(tour, len(tour), -1)
                    self.tours[tour_idx] = {
                        'tour': tour,
                        'node_pos': tour_nodes_to_node_rep(tour),
                        'cum_dems': tour_nodes_to_cum_demands(tour, self.node_demands)
                    }
        self.nbh = self.make_nbh()

    def all_tours_as_list(self, remove_last_depot=False, remove_first_depot=False):
        all_tours_list = [t['tour'] for t in self.tours.values()]
        if remove_last_depot:
            all_tours_list = [t[:-1] for t in all_tours_list]
        if remove_first_depot:
            all_tours_list = [t[1:] for t in all_tours_list]
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
        # no need to update tour_idx here

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
        if dst_tour_idx == -1:
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
            # this shouldn't be neccessary but paranoid
        else:
            assert np.all(src_tour_nodes_new == np.array([0, -1])), "just double checking"

        self.tours[dst_tour_idx] = {
            'tour': dst_tour_nodes_new,
            'node_pos': dst_tour_nodes_pos_new,
            'cum_dems': tour_nodes_to_cum_demands(dst_tour_nodes_new, self.node_demands)
        }

    def make_nbh(self):
        return VRPNbHAutoReg(self)

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


def get_normalized_nbh_rep(_src_edges: List[np.ndarray], _dst_edges: List[np.ndarray]):
    src_edges = list(filter(lambda e: e[0] != e[1], [normalize_edge(e) for e in _src_edges]))
    dst_edges = list(filter(lambda e: e[0] != e[1], [normalize_edge(e) for e in _dst_edges]))
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

    assert np.isclose(tour_1_new_dem + tour_0_new_dem, C0 + C1), \
        f"{cum_demand_0}, {cum_demand_1}, {tour_0_new_dem}, {tour_1_new_dem}, {C0}, {C1}"
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


def normalize_edges(edges: np.array):
    """
    assume edges of the shape ... x 2 where last axis is u, v
    :param edges:
    :return:
    """
    edges_norm = edges.copy()
    edges_norm[edges_norm == -1] = 0
    edges_norm = np.sort(edges_norm, axis=-1)
    return edges_norm


def flatten_deduplicate_reloc_nbh(
    reloc_nbhs: List[Dict[str, Union[np.ndarray, int]]],
    state: VRPState
):
    reloc_nbhs_dict = {}
    for nbh in reloc_nbhs:
        tour0 = nbh['src_tour']
        tour1 = nbh['dst_tour']
        node_edge_pairs = nbh['node_edges_pairs']
        for node_edge_pair in node_edge_pairs:
            src_node = node_edge_pair[0]
            src_u = node_edge_pair[1:3]
            src_v = node_edge_pair[3:5]
            dst_w = node_edge_pair[5:7]
            src_edges = [src_u, src_v, dst_w]
            dst_wp = np.array([src_u[0], src_v[1]])
            src_up = np.array([dst_w[0], src_node])
            src_vp = np.array([src_node, dst_w[1]])
            dst_edges = [
                dst_wp,
                src_up,
                src_vp,
            ]
            nbh_norm = get_normalized_nbh_rep(src_edges, dst_edges)
            e_add, e_rem = nbh_norm
            no_op = len(set(e_add).symmetric_difference(set(e_rem))) == 0

            if (tour1 is not None) and (not check_relocate_move_valid(
                    src_node,
                    state.tours[tour1]['cum_dems'],
                    state.node_demands,
                    max_tour_dem=state.max_tour_demand
            )):
                continue

            ew = state.edge_weights
            cost = 0
            for e in e_add:
                cost -= ew[e[0], e[1]]
            for e in e_rem:
                cost += ew[e[0], e[1]]
            # remove no_ops and duplicates
            if not no_op and (nbh_norm not in reloc_nbhs_dict):
                nb = {
                    "nb_type": "reloc",
                    "tour0": tour0,
                    "tour1": tour1 if tour1 is not None else -1,
                    "src_node": src_node,
                    "src_u": src_u,
                    "src_v": src_v,
                    "dst_w": dst_w,
                    "dst_wp": dst_edges[0],
                    "src_up": dst_edges[1],
                    "src_vp": dst_edges[2],
                    "cost": cost
                }
                reloc_nbhs_dict[nbh_norm] = nb

    # import random
    # if random.uniform(0, 1.) < 0.05:
    #     print(reloc_nbhs)
    #     print(state.tours)
    #     print(state.edge_weights)
    #     import pickle
    #     import uuid
    #     with open(f"example_flatten_reloc_inputs/dummy_flatten_deduplicate_reloc_nbh_inputs_{uuid.uuid4()}.pkl",
    #               "wb") as fp:
    #         pickle.dump({
    #             'reloc_nbhs': reloc_nbhs,
    #             'state.node_demands': state.node_demands,
    #             'state.tours': state.tours,
    #             'state.edge_weights': state.edge_weights,
    #             'state.max_tour_demand': state.max_tour_demand,
    #             'deduped_reloc': reloc_nbhs_dict
    #         }, fp)
    return reloc_nbhs_dict


def flatten_deduplicate_cross_nbh(
    cross_nbhs: List[Dict[str, Union[np.ndarray, int]]],
    state: VRPState
):
    cross_nbh_dict = {}
    for nbh in cross_nbhs:
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

            e_add, e_rem = nbh_norm
            no_op = len(set(e_add).symmetric_difference(set(e_rem))) == 0

            ew = state.edge_weights
            cost = 0
            for e in e_add:
                cost -= ew[e[0], e[1]]
            for e in e_rem:
                cost += ew[e[0], e[1]]
            # remove no_ops and duplicates
            if (
                    not no_op and
                    (nbh_norm not in cross_nbh_dict) and
                    check_cross_move_valid(e0, e1, e0p, e1p, cum_dem_0, cum_dem_1, state.max_tour_demand)
            ):
                nb = {
                    "nb_type": "cross",
                    "tour0": tour0,
                    "tour1": tour1,
                    "e0": e0,
                    "e1": e1,
                    "e0p": e0p,
                    "e1p": e1p,
                    "cost": cost
                }
                cross_nbh_dict[nbh_norm] = nb
            # second move
            e0p = edge_pair[8:10]
            e1p = edge_pair[10:12]
            nbh_norm = get_normalized_nbh_rep([e0, e1], [e0p, e1p])

            e_add, e_rem = nbh_norm
            no_op = len(set(e_add).symmetric_difference(set(e_rem))) == 0

            cost = 0
            for e in e_add:
                cost -= ew[e[0], e[1]]
            for e in e_rem:
                cost += ew[e[0], e[1]]

            # remove no_ops and duplicates
            if (
                    not no_op and
                    (nbh_norm not in cross_nbh_dict) and
                    check_cross_move_valid(e0, e1, e0p, e1p, cum_dem_0, cum_dem_1, state.max_tour_demand)
            ):
                nb = {
                    "nb_type": "cross",
                    "tour0": tour0,
                    "tour1": tour1,
                    "e0": e0,
                    "e1": e1,
                    "e0p": e0p,
                    "e1p": e1p,
                    "cost": cost
                }
                cross_nbh_dict[nbh_norm] = nb

    # import random
    # if random.uniform(0, 1.) < 0.01:
    #     print(cross_nbhs)
    #     print(state.node_demands)
    #     print(state.tours)
    #     print(state.edge_weights)
    #     print(state.max_tour_demand)
    #     import pickle
    #     import uuid
    #     with open(f"example_flatten_cross_inputs/dummy_flatten_deduplicate_cross_nbh_inputs_{uuid.uuid4()}.pkl",
    #               "wb") as fp:
    #         pickle.dump({
    #             'cross_nbhs': cross_nbhs,
    #             'state.node_demands': state.node_demands,
    #             'state.tours': state.tours,
    #             'state.edge_weights': state.edge_weights,
    #             'state.max_tour_demand': state.max_tour_demand,
    #             'deduped_cross': cross_nbh_dict
    #         }, fp)

    return cross_nbh_dict


def flatten_deduplicate_2opt_nbh(
    twoopt_nbhs: List[Dict[str, Union[np.ndarray, int]]],
    state: VRPState
):
    two_opt_nbh_dict = {}
    for nbh in twoopt_nbhs:
        tour_idx = nbh['tour_idx']
        tour_edges_pairs = nbh['tour_edges_pairs']
        for tour_edge_pair in tour_edges_pairs:
            e0 = tour_edge_pair[:2]
            e1 = tour_edge_pair[2:4]
            e0p = tour_edge_pair[4:6]
            e1p = tour_edge_pair[6:8]
            nbh_norm = get_normalized_nbh_rep([e0, e1], [e0p, e1p])
            e_add, e_rem = nbh_norm
            no_op = len(set(e_add).symmetric_difference(set(e_rem))) == 0

            ew = state.edge_weights
            cost = 0
            for e in e_add:
                cost -= ew[e[0], e[1]]
            for e in e_rem:
                cost += ew[e[0], e[1]]

            # remove no_ops and duplicates
            if not no_op and (nbh_norm not in two_opt_nbh_dict):
                nb = {
                    "nb_type": "2opt",
                    "tour_idx": tour_idx,
                    "e0": e0,
                    "e1": e1,
                    "e0p": e0p,
                    "e1p": e1p,
                    "cost": cost
                }
                two_opt_nbh_dict[nbh_norm] = nb

    # import random
    # if random.uniform(0, 1.) < 0.01:
    #     print(twoopt_nbhs)
    #     print(state.tours)
    #     print(state.edge_weights)
    #     import pickle
    #     import uuid
    #     with open(f"example_flatten_twoopt_inputs/dummyoopt_nbh_inputs_{uuid.uuid4()}.pkl",
    #               "wb") as fp:
    #         pickle.dump({
    #             'twoopt_nbhs': twoopt_nbhs,
    #             'state.tours': state.tours,
    #             'state.edge_weights': state.edge_weights,
    #             'deduped_cross': two_opt_nbh_dict
    #         }, fp)

    return two_opt_nbh_dict


class VRPNbHAutoReg:
    @staticmethod
    def vectorize_moves(reloc_nbh=None, cross_nbh=None, twp_opt_nbh=None):
        if reloc_nbh is None:
            reloc_nbh = []
        if cross_nbh is None:
            cross_nbh = []
        if twp_opt_nbh is None:
            twp_opt_nbh = []

        reloc_nbh_vect = vectorize_reloc_moves(reloc_nbh)
        cross_nbh_vect = vectorize_cross_moves(cross_nbh)
        twp_opt_nbh_vect = vectorize_twopt_moves(twp_opt_nbh)

        return reloc_nbh_vect, cross_nbh_vect, twp_opt_nbh_vect

    def _get_first_move_nbh(self):
        first_move_nbh = []
        edges_vect = []
        nodes_vect = []
        for tour_idx, tour_nodes in self.tour_nodes.items():
            _tour_nodes = tour_nodes[1:-1]
            nodes_vect.append(_tour_nodes)  # we don't want depot nodes here
            first_move_nbh.extend(
                [{'type': 'node', 'node': n, 'tour_idx': tour_idx} for n in _tour_nodes]
            )
        for tour_idx, tour_edges in self.tour_edges.items():
            edges_vect.append(tour_edges)
            first_move_nbh.extend(
                [{'type': 'edge', 'edge': e, 'tour_idx': tour_idx} for e in tour_edges]
            )

        edges_vect = np.concatenate(edges_vect, axis=0)  # num_edges x 2
        edges_vect = normalize_edges(edges_vect)
        nodes_vect = np.concatenate(nodes_vect, axis=0)  # num_nodes

        return edges_vect, nodes_vect, first_move_nbh

    def _get_second_move_nbh(
        self,
        state: VRPState,
        moves_0,
        actions_top_k_0: np.ndarray  # k
    ):
        reloc_nbhs = []
        cross_nbhs = []
        twoopt_nbhs = []
        second_moves = []
        selected_first_actions_k = []
        for i, (action_0, move_0) in enumerate(zip(actions_top_k_0, moves_0)):
            _second_moves = []
            if move_0['type'] == 'node':
                node = move_0['node']
                node_tour = move_0['tour_idx']
                node_pos = self.tours[node_tour]['node_pos'][node]
                _reloc_nbh = enumerate_relocate_neighborhood_given(
                    node, node_tour, node_pos, self.tour_edges
                )
                _reloc_nbh = flatten_deduplicate_reloc_nbh(_reloc_nbh, state=state)
                # _reloc_nbh = vrpstate.flatten_reloc_nbh(
                #     _reloc_nbh,
                #     state.tours,
                #     state.node_demands,
                #     state.edge_weights,
                #     state.max_tour_demand
                # )
                _reloc_nbh = list(_reloc_nbh.values())
                reloc_nbhs.extend(_reloc_nbh)
                _second_moves.extend(_reloc_nbh)
                # reloc_nbh_vect, cross_nbh_vect, twp_opt_nbh_vect = VRPNbHAutoReg.vectorize_moves(_reloc_nbh, [], [])
            elif move_0['type'] == 'edge':
                edge = move_0['edge']
                edge_tour = move_0['tour_idx']
                _cross_nbh = enumerate_cross_neighborhood_given(edge_tour, edge, self.tour_edges)
                _two_opt_nbh = enumerate_2_opt_neighborhood_given(edge_tour, edge, self.tour_edges)
                _cross_nbh = flatten_deduplicate_cross_nbh(cross_nbhs=_cross_nbh, state=state)
                _two_opt_nbh = flatten_deduplicate_2opt_nbh(_two_opt_nbh, state=state)
                # _cross_nbh = vrpstate.flatten_cross_nbh(
                #     _cross_nbh,
                #     state.tours,
                #     state.node_demands,
                #     state.edge_weights,
                #     state.max_tour_demand
                # )
                # _two_opt_nbh = vrpstate.flatten_2opt_nbh(
                #     _two_opt_nbh,
                #     state.tours,
                #     state.edge_weights,
                # )

                _cross_nbh = list(_cross_nbh.values())
                _two_opt_nbh = list(_two_opt_nbh.values())
                cross_nbhs.extend(_cross_nbh)
                twoopt_nbhs.extend(_two_opt_nbh)
                _second_moves.extend(_cross_nbh + _two_opt_nbh)
            selected_first_actions_k.append(
                np.full(shape=(len(_second_moves)), fill_value=i, dtype=int)
            )
            second_moves.extend(_second_moves)

        selected_first_actions_k = np.concatenate(selected_first_actions_k, axis=0)
        reloc_nbh_vect, cross_nbh_vect, twp_opt_nbh_vect = VRPNbHAutoReg.vectorize_moves(reloc_nbhs, cross_nbhs, twoopt_nbhs)
        return reloc_nbh_vect, cross_nbh_vect, twp_opt_nbh_vect, second_moves, selected_first_actions_k

    def __init__(self, state: VRPState):
        self.tour_edges, _ = enumerate_all_tours_edges(
            state.tour_idx_to_tour(), directed=True
        )
        self.tour_nodes = enumerate_all_tours_nodes(state.tour_idx_to_tour())
        self.tours = state.tours

        self.edges_vect, self.nodes_vect, self.first_move_nbh = self._get_first_move_nbh()

        self.reloc_nbh_vect = None
        self.cross_nbh_vect = None
        self.twp_opt_nbh_vect = None
        self.second_moves = None
        self.selected_first_actions_k = None


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


def get_node_embs(x_emb: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    """
    :param e_emb: b x v x v x h - edge embeddings
    :param edges: ... x 3 - (batch_idx, u, v)
    :return: ... x h
    """
    bs = nodes[..., 0]
    us = nodes[..., 1]
    ret = x_emb[bs, us]
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
            vectorized[key] = np.zeros(shape=(len(cross_moves), 2), dtype=int)
            for i, cross_move in enumerate(cross_moves):
                edge = cross_move[key]
                e0, e1 = edge
                if e0 == -1:
                    e0 = 0
                if e1 == -1:
                    e1 = 0
                if e0 > e1:
                    e0, e1 = e1, e0
                vectorized[key][i, 0] = e0
                vectorized[key][i, 1] = e1
            # vectorized[key] = np.stack([
            #     normalize_edge(np.array(cross_move[key]))
            #     for cross_move in cross_moves
            # ], axis=0)
    vectorized['cost'] = np.array(
        [reloc_move['cost'] for reloc_move in cross_moves]
    )[:, None]
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
            vectorized[key] = np.zeros(shape=(len(reloc_moves), 2), dtype=int)
            for i, reloc_move in enumerate(reloc_moves):
                edge = reloc_move[key]
                e0, e1 = edge
                if e0 == -1:
                    e0 = 0
                if e1 == -1:
                    e1 = 0
                if e0 > e1:
                    e0, e1 = e1, e0
                vectorized[key][i, 0] = e0
                vectorized[key][i, 1] = e1
            # vectorized[key] = np.stack([
            #     normalize_edge(np.array(reloc_move[key]))
            #     for reloc_move in reloc_moves
            # ], axis=0)
    vectorized['cost'] = np.array(
        [reloc_move['cost'] for reloc_move in reloc_moves]
    )[:, None]
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
            vectorized[key] = np.zeros(shape=(len(twopt_moves), 2), dtype=int)
            for i, twopt_move in enumerate(twopt_moves):
                edge = twopt_move[key]
                e0, e1 = edge
                if e0 == -1:
                    e0 = 0
                if e1 == -1:
                    e1 = 0
                if e0 > e1:
                    e0, e1 = e1, e0
                vectorized[key][i, 0] = e0
                vectorized[key][i, 1] = e1
            # vectorized[key] = np.stack([
            #     normalize_edge(np.array(twopt_move[key]))
            #     for twopt_move in twopt_moves
            # ], axis=0)
    vectorized['cost'] = np.array(
        [reloc_move['cost'] for reloc_move in twopt_moves]
    )[:, None]
    return vectorized


def embed_cross_heuristic(cross_moves_vectorized, edge_embeddings: torch.Tensor, cross_move_mlp: torch.nn.Module, cost_mlp: torch.nn.Module):
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
    cost = []
    num_moves = []
    for i, m in enumerate(cross_moves_vectorized):
        num_moves_i = m['e0'].shape[0]
        batch_index = i * np.ones(shape=(num_moves_i, 1))
        e0.append(np.concatenate([batch_index, m['e0']], axis=1))
        e1.append(np.concatenate([batch_index, m['e1']], axis=1))
        e0p.append(np.concatenate([batch_index, m['e0p']], axis=1))
        e1p.append(np.concatenate([batch_index, m['e1p']], axis=1))
        cost.append(m['cost'])
        num_moves.append(num_moves_i)

    # sum_b ( num_moves(b) )  x  3
    e0 = torch.as_tensor(np.concatenate(e0, axis=0)).long()
    e1 = torch.as_tensor(np.concatenate(e1, axis=0)).long()
    e0p = torch.as_tensor(np.concatenate(e0p, axis=0)).long()
    e1p = torch.as_tensor(np.concatenate(e1p, axis=0)).long()
    cost = torch.as_tensor(np.concatenate(cost, axis=0)).float().to(edge_embeddings.device)
    # sum_b ( num_moves(b) ) x 1

    edges = torch.stack([e0, e1, e0p, e1p], dim=0)
    edges_emb = get_edge_embs(edge_embeddings, edges, symmetric=True)
    # sum_b ( num_moves(b) )  x  3
    e0_emb, e1_emb, e0p_emb, e1p_emb = edges_emb[0], edges_emb[1], edges_emb[2], edges_emb[3]
    cost_emb = cost_mlp(cost)

    # MLP the embeddings into a single embedding
    # sum_b ( num_moves(b) ) x (5 * h)
    cross_move_input_emb = torch.cat([e0_emb, e1_emb, e0p_emb, e1p_emb, cost_emb], dim=1)
    # MLP input dim is 5 x h output dim is h
    # sum_b ( num_moves(b) ) x (5 * h)
    cross_move_embeddings = cross_move_mlp(cross_move_input_emb)
    # split it into a list of tensors using num_moves
    assert cross_move_embeddings.shape[0] == sum(num_moves)
    cross_move_embeddings = torch.split(cross_move_embeddings, num_moves, dim=0)
    # cat it with 2opt and reloc
    return cross_move_embeddings


def embed_reloc_heuristic(reloc_moves_vectorized, edge_embeddings: torch.Tensor, reloc_move_mlp: torch.nn.Module, cost_mlp: torch.nn.Module):
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
    cost = []
    for i, m in enumerate(reloc_moves_vectorized):
        num_moves_i = m['src_u'].shape[0]
        batch_index = i * np.ones(shape=(num_moves_i, 1))
        src_u.append(np.concatenate([batch_index, m['src_u']], axis=1))
        src_v.append(np.concatenate([batch_index, m['src_v']], axis=1))
        dst_w.append(np.concatenate([batch_index, m['dst_w']], axis=1))
        src_up.append(np.concatenate([batch_index, m['src_up']], axis=1))
        src_vp.append(np.concatenate([batch_index, m['src_vp']], axis=1))
        dst_wp.append(np.concatenate([batch_index, m['dst_wp']], axis=1))
        cost.append(m['cost'])
        num_moves.append(num_moves_i)

    # sum_b ( num_moves(b) )  x  3
    src_u = torch.as_tensor(np.concatenate(src_u, axis=0)).long()
    src_v = torch.as_tensor(np.concatenate(src_v, axis=0)).long()
    dst_w = torch.as_tensor(np.concatenate(dst_w, axis=0)).long()
    src_up = torch.as_tensor(np.concatenate(src_up, axis=0)).long()
    src_vp = torch.as_tensor(np.concatenate(src_vp, axis=0)).long()
    dst_wp = torch.as_tensor(np.concatenate(dst_wp, axis=0)).long()
    cost = torch.as_tensor(np.concatenate(cost, axis=0)).float().to(edge_embeddings.device)
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
    cost_emb = cost_mlp(cost)

    # MLP the embeddings into a single embedding
    # sum_b ( num_moves(b) ) x (6 * h)
    reloc_move_input_emb = torch.cat([src_u_emb, src_v_emb, dst_w_emb, src_up_emb, src_vp_emb, dst_wp_emb, cost_emb], dim=1)
    # MLP input dim is 4 x h output dim is h
    # sum_b ( num_moves(b) ) x (4 * h)
    reloc_move_embeddings = reloc_move_mlp(reloc_move_input_emb)
    # split it into a list of tensors using num_moves
    assert reloc_move_embeddings.shape[0] == sum(num_moves)
    reloc_move_embeddings = torch.split(reloc_move_embeddings, num_moves, dim=0)
    # cat it with 2opt and reloc
    return reloc_move_embeddings


from gym import Env
import copy
import cloudpickle


def worker(remote, parent_remote, env_fn, env_idx):
    parent_remote.close()
    env: VRPEnvBase = env_fn()
    env.init()
    np.random.seed(env_idx)

    cur_instance = None
    cur_instance_id = None
    max_num_steps = None

    while True:
        cmd, data = remote.recv()

        if cmd == 'step':
            action = data
            ob, reward, done = env.step(action)
            remote.send((ob, reward, done))

        # elif cmd == 'reset':
        #     remote.send(env.reset())

        elif cmd == 'reset_episode':
            assert cur_instance_id, "cannot reset episode before setting a run instance"
            remote.send(env.set_instance_as_state(
                instance=env.cur_instance,
                init_tour=env.state.all_tours_as_list(remove_last_depot=True, remove_first_depot=True),
                best_tour=env.best_state.all_tours_as_list(remove_last_depot=True, remove_first_depot=True),
                id=cur_instance_id,
                max_num_steps=max_num_steps
            ))
        
        elif cmd == 'set_instance_run':
            cur_instance, cur_instance_id, max_num_steps = data
            remote.send(env.set_instance_as_state(
                instance=cur_instance,
                init_tour=None,
                best_tour=None,
                id=cur_instance_id,
                max_num_steps=max_num_steps
            ))

        elif cmd == 'get_state':
            remote.send(env.get_state())

        elif cmd == 'get_instance':
            remote.send(env.cur_instance)

        elif cmd == 'get_second_move':
            move_0, action_0 = data
            remote.send(env.get_second_move(move_0, action_0))

        # elif cmd == 'render':
        #     remote.send(env.render())

        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError


import pickle
from multiprocessing import Pipe, Process


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)

    def __call__(self):
        return self.x()


def make_mp_envs(num_env, num_steps, max_tour_demand, reward_mode, initializer):
    def make_env():
        def fn():
            env = VRPEnvBase(
                max_num_steps=num_steps,
                max_tour_demand=max_tour_demand,
                reward_mode=reward_mode,
                initializer=initializer
            )
            return env
        return fn
    return SubprocVecEnv([make_env() for i in range(num_env)])


class SubprocVecEnv:
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.no_of_envs = len(env_fns)
        self.remotes, self.work_remotes = \
            zip(*[Pipe() for _ in range(self.no_of_envs)])
        self.ps = []

        for env_idx, (wrk, rem, fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns)):
            proc = Process(target=worker,
                           args=(wrk, rem, CloudpickleWrapper(fn), env_idx))
            self.ps.append(proc)

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        if self.waiting:
            raise Exception
        self.waiting = True

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', (action)))

    def step_wait(self):
        if not self.waiting:
            raise Exception
        self.waiting = False

        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones = zip(*results)
        return obs, rews, dones

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    
    def set_instances_run(
        self,
        instances, instance_ids, max_num_stepss
    ):
        # print(instance_ids)
        for remote, instance, instance_id, max_num_steps in \
                zip(self.remotes, instances, instance_ids, max_num_stepss):
            remote.send(('set_instance_run', (instance, instance_id, max_num_steps)))
        return [remote.recv() for remote in self.remotes]
    
    def reset_episode(
        self,
    ):
        """
        resets the step counter (and starts a new episode),
        but otherwise keeps the state exactly the same
        :return:
        """
        for remote in self.remotes:
            remote.send(('reset_episode', (None, )))
        return [remote.recv() for remote in self.remotes]

    def get_state(self):
        for remote in self.remotes:
            remote.send(('get_state', (None, )))
        states = [remote.recv() for remote in self.remotes]
        return states

    def get_first_move_from_states(self):
        pass

    def get_second_move_from_states(self, moves_0, actions_0):
        for remote, move_0, action_0 in zip(self.remotes, moves_0, actions_0):
            remote.send(('get_second_move', (move_0, action_0)))
        second_moves = [remote.recv() for remote in self.remotes]
        return second_moves

    def get_instance(self):
        for remote in self.remotes:
            remote.send(('get_instance', (None,)))
        states = [remote.recv() for remote in self.remotes]
        return states

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class VRPReward(Enum):
    FINAL_COST = 'FINAL_COST'
    DELTA_COST = 'DELTA_COST'


class VRPEnvBase(Env):
    def __init__(
        self,
        reward_mode: VRPReward,
        max_num_steps=50,
        ret_best_state=True,
        max_tour_demand=10.,
        initializer: VRPInitTour = VRPInitTour.SINGLETON
    ):
        super(VRPEnvBase, self).__init__()
        # config vars
        self.max_num_steps = max_num_steps
        self.ret_best_state = ret_best_state
        self.max_tour_demand = max_tour_demand
        # this is the original representation of the problem before any fudging
        self.cur_instance = None
        self.reward_mode = reward_mode
        self.initializer = initializer

    def init(self):
        self.cur_step = -1

    def _make_state_from_batch_and_tour(self, b, init_tours, id):
        # normalize it so that in the VRPState, max_tour_demand == VRPState.VEHICLE_CAPACITY
        scale = VRPState.VEHICLE_CAPACITY / self.max_tour_demand
        return VRPState(
            nodes_coord=b['nodes_coord'],
            node_demands=b['demands'] * scale,
            tours_init=init_tours,
            max_tour_demand=VRPState.VEHICLE_CAPACITY,
            # opt_tour=b['tour_nodes'][0],  # TODO: optimal tour
            id=id,
            opt_tour_dist=b.get('opt_dist'),
            init_tour=self.initializer
        )

    def set_instance_as_state(
        self,
        instance,
        init_tour=None,
        best_tour=None,
        id: Optional[int] = None,
        max_num_steps: Optional[int] = None,
        ret_opt_tour: bool = False
    ):
        b = instance
        state = self._make_state_from_batch_and_tour(b, init_tour, id)
        best_state = None
        if best_tour is not None:
            best_state = self._make_state_from_batch_and_tour(b, best_tour, id)
        self._set_state(
            instance,
            state=state,
            best_state=best_state,
            max_num_steps=max_num_steps
        )
        return self.get_state()

    def _set_state(
        self,
        instance: Dict[str, Any],
        state: VRPState,
        best_state: Optional[VRPState] = None,
        max_num_steps: Optional[int] = None
    ):
        self.cur_instance = copy.deepcopy(instance)
        self.state = copy.deepcopy(state)
        self.best_state = copy.deepcopy(self.state) if best_state is None else best_state
        self.cur_step = 0
        self.done = False
        # option to reset the episode len
        if max_num_steps is not None:
            self.max_num_steps = max_num_steps

    def get_state(self):
        if self.ret_best_state:
            return (self.state, self.best_state)
        return self.state

    def get_second_move(self, move_0, action_0):
        nbh = self.state.get_nbh()
        return nbh._get_second_move_nbh(self.state, move_0, action_0)

    def step(self, action: Dict):
        if self.done:
            return self.state, 0, self.done

        self.cur_step += 1
        if action['terminate']:
            self.done = True

        if self.cur_step == self.max_num_steps:
            self.done = True

        delta = self.best_state.get_cost(exclude_depot=False)
        if self.reward_mode == VRPReward.FINAL_COST:
            if self.done:
                # if the action given in t-1 was terminate, or cur_step == T
                # then the reward is cost(S[t-1])
                reward = -self.best_state.get_cost(exclude_depot=False)
            else:
                move = action['move']
                self.state.apply_move(move)
                reward = 0.
                if self.state.get_cost(exclude_depot=False) < self.best_state.get_cost(exclude_depot=False):
                    self.best_state = copy.deepcopy(self.state)
        else:
            if not self.done:
                move = action['move']
                self.state.apply_move(move)
                if self.state.get_cost(exclude_depot=False) < self.best_state.get_cost(exclude_depot=False):
                    self.best_state = copy.deepcopy(self.state)
            delta -= self.best_state.get_cost(exclude_depot=False)
            reward = delta

        return self.get_state(), reward, self.done


class VRPEnvRandom(VRPEnvBase):

    def get_next_instance(self):
        N = self.num_nodes
        # get a new batch from TSPReader and initialize the state
        demands = np.ones(shape=(N))
        coords = np.zeros(shape=(N + 1, 2))
        coords[0] = 0.5
        coords[1:] = np.random.random(size=(N, 2))

        instance = {
            'nodes_coord': coords, 'demands': demands
        }
        self.last_instance_id += 1
        return instance

    def init(self):
        super().init()
        self.rng = np.random.default_rng(self.seed)

    def __init__(
        self,
        reward_mode: VRPReward,
        num_nodes=10,
        max_num_steps=10,
        ret_best_state=True,
        max_tour_demand=10.,
        ret_opt_tour=False,
        initializer=VRPInitTour.SINGLETON,
        seed=42
    ):
        super().__init__(
            max_num_steps=max_num_steps,
            ret_best_state=ret_best_state,
            max_tour_demand=max_tour_demand,
            initializer=initializer,
            reward_mode=reward_mode,
        )
        # config vars
        self.num_nodes = num_nodes
        self.last_instance_id = -1
        self.seed = seed
        self.ret_opt_tour = ret_opt_tour
        self.init()

    def reset_episode(self):
        """
        similar to reset(), but does not resample the node tour - re-uses the state from last episode, never fetches
        new instance
        :return:
        """
        self.set_instance_as_state(
            self.cur_instance,
            init_tour=self.state.all_tours_as_list(remove_last_depot=True, remove_first_depot=True),
            best_tour=self.best_state.all_tours_as_list(remove_last_depot=True, remove_first_depot=True),
            id=self.state.id,
            ret_opt_tour=self.ret_opt_tour
        )
        return self.get_state()

    def reset(self, fetch_next=True, max_num_steps=None):
        """
        resets the *run* - fetches a new instance (if fetch_next=True), and resamples a node tour from random
        :param fetch_next:
        :return:
        """
        if fetch_next or self.cur_instance is None:
            b = self.get_next_instance()
        else:
            b = self.cur_instance
        self.set_instance_as_state(
            b,
            init_tour=None,
            best_tour=None,
            id=self.last_instance_id,
            ret_opt_tour=self.ret_opt_tour,
            max_num_steps=max_num_steps
        )
        return self.get_state()

    def get_second_moves(self, move_0, action_0):
        return [self.get_second_move(move_0[0], action_0[0])]


class VRPMultiEnvAbstract(Env):
    def __init__(
        self,
        reward_mode: VRPReward,
        num_nodes=10,
        max_num_steps=50,
        max_tour_demand=10.,
        ret_opt_tour=False,
        num_samples_per_instance=1,
        num_instance_per_batch=1,
        seed=42,
        initializer=VRPInitTour.SINGLETON
    ):
        self.num_envs = num_samples_per_instance * num_instance_per_batch
        self.max_num_steps = max_num_steps
        self.max_tour_demand = max_tour_demand
        self.envs = make_mp_envs(
            num_env=self.num_envs,
            num_steps=max_num_steps,
            max_tour_demand=max_tour_demand,
            reward_mode=reward_mode,
            initializer=initializer
        )

        self.num_nodes = num_nodes
        self.seed = seed
        assert num_samples_per_instance > 0
        self.num_samples_per_instance = num_samples_per_instance
        self.num_instance_per_batch = num_instance_per_batch
        self.last_instance_id = -1
        self.ret_opt_tour = ret_opt_tour
        self.init()

    def init(self):
        self.cur_instances = [None for _ in range(self.num_samples_per_instance)]
        self.cur_instance_ids = [None for _ in range(self.num_samples_per_instance)]
        self.rng = np.random.default_rng(self.seed)
    
    def reset_episode(self):
        """
        similar to reset(), but does not resample the node tour - re-uses the state from last episode, never fetches
        new instance
        :return:
        """
        self.envs.reset_episode()

    def get_next_instance(self):
        raise NotImplementedError()

    def reset(self, fetch_next=True, max_num_steps=None):
        if fetch_next or self.cur_instances[0] is None:
            instances = []
            instance_ids = []
            for i in range(self.num_instance_per_batch):
                instances.append(self.get_next_instance())
                instance_ids.append(self.last_instance_id)
                for j in range(1, self.num_samples_per_instance):
                    instances.append(copy.deepcopy(instances[-1]))
                    instance_ids.append(instance_ids[-1])
            self.cur_instances = instances
            self.cur_instance_ids = instance_ids

        max_num_steps = max_num_steps if max_num_steps else self.max_num_steps
        return self.envs.set_instances_run(
            self.cur_instances,
            self.cur_instance_ids,
            [max_num_steps] * self.num_envs
        )

    def step(self, actions):
        assert len(actions) == self.num_envs
        return self.envs.step(actions)

    def get_state(self):
        return self.envs.get_state()

    def get_instance(self):
        return self.envs.get_instance()

    def get_second_moves(self, moves_0, actions_0: np.ndarray):
        return self.envs.get_second_move_from_states(moves_0, actions_0)


class VRPMultiRandomEnv(VRPMultiEnvAbstract):
    def get_next_instance(self):
        N = self.num_nodes
        # get a new batch from TSPReader and initialize the state
        demands = np.ones(shape=(N))
        coords = np.zeros(shape=(N + 1, 2))
        coords[0] = 0.5
        coords[1:] = np.random.random(size=(N, 2))

        instance = {
            'nodes_coord': coords, 'demands': demands
        }
        self.last_instance_id += 1
        return instance


import pickle
import random
class VRPMultiFileEnv(VRPMultiEnvAbstract):
    def __init__(self, data_f, results_f=None, *args, **kwargs):
        self.data_f = data_f
        self.results_f = results_f
        super().__init__(*args, **kwargs)
        self.init()

    def init(self):
        with open(self.data_f, 'rb') as fp:
            self.data = pickle.load(fp)
        random.shuffle(self.data)
        self.data_iter = self.data.__iter__()

    def get_next_instance(self):
        try:
            instance = next(self.data_iter)
        except StopIteration as e:
            random.shuffle(self.data)
            self.data_iter = self.data.__iter__()
            instance = next(self.data_iter)

        # format it properly
        depot = np.array(instance['depot'])
        nodes = np.array(instance['nodes'])
        capacity = instance['capacity']
        opt_dist = instance.get('opt_dist')
        opt_tour = instance.get('opt_tour')
        if len(nodes) != self.num_nodes:
            raise ValueError("Num nodes do not match")
        if capacity != self.max_tour_demand:
            raise ValueError("max capacity doesn't match")
        # demands are assumed to be 1
        demands = np.ones(shape=(self.num_nodes))
        coords = np.concatenate((depot[None, :], nodes), axis=0)
        instance = {
            'nodes_coord': coords, 'demands': demands, 'opt_dist': opt_dist, 'opt_tour': opt_tour
        }
        self.last_instance_id += 1
        return instance



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
            # ax1.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)
            for i, x, y in zip(r, xs, ys):
                ax1.plot(x, y, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)
                ax1.text(x, y, s=str(i))

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


def plot_state(state: VRPState, fname: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(4, 4))
    data = {'loc': state.nodes_coord[1:], 'depot': state.nodes_coord[0], 'demand': state.get_node_demands()}
    plot_vehicle_routes(data, [t[1:] for t in state.all_tours_as_list(remove_last_depot=True)], ax1=ax)
    if fname:
        plt.savefig(fname)
    plt.show()


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

    tour_edges, _ = enumerate_all_tours_edges(
        state.tour_idx_to_tour(), directed=True
    )
    reloc_nbhs = enumerate_relocate_neighborhood(state.tour_idx_to_tour(), tour_edges)
    nbh_vect = []

    for nbh in reloc_nbhs:
        tour0 = nbh['src_tour']
        tour1 = nbh['dst_tour']
        node_edge_pairs = nbh['node_edges_pairs']
        nbh_vect.append(node_edge_pairs)

    nbh_vect = np.concatenate(nbh_vect, axis=0)
    print(nbh_vect)


    src_u = nbh_vect[:, 1:3]
    src_v = nbh_vect[:, 3:5]
    dst_w = nbh_vect[:, 5:7]
    src_node = nbh_vect[:, 0]

    src_edges = np.stack([src_u, src_v, dst_w], axis=1)
    dst_edges = np.stack([
        np.stack([src_u[:, 0], src_v[:, 1]], axis=1),
        np.stack([dst_w[:, 0], src_node], axis=1),
        np.stack([src_node, dst_w[:, 1]], axis=1)
    ], axis=1)

    print(dst_edges.shape, src_edges.shape)

    nbh_rep_vectorized = dedup_nbh_vectorized(src_edges, dst_edges)

    assert False

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


