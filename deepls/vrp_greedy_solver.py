from deepls.VRPState import (
    enumerate_all_tours_edges,
    enumerate_all_tours_nodes,
    flatten_deduplicate_reloc_nbh,
    flatten_deduplicate_cross_nbh,
    flatten_deduplicate_2opt_nbh,
    VRPState
)
from typing import Dict, List
import numpy as np


class VRPNbH:
    @staticmethod
    def enumerate_all_nbs(state: VRPState):
        # relocation heuristic
        tour_edges, _ = enumerate_all_tours_edges(
            state.tour_idx_to_tour(), directed=True
        )
        reloc_nbhs = enumerate_relocate_neighborhood(state.tour_idx_to_tour(), tour_edges)
        reloc_nbhs_dict = flatten_deduplicate_reloc_nbh(reloc_nbhs, state)

        # cross heuristic
        cross_nbhs = enumerate_cross_heuristic_neighborhood(tour_edges)
        cross_nbh_dict = flatten_deduplicate_cross_nbh(cross_nbhs, state)

        # 2opt heuristics
        twoopt_nbhs = enumerate_2_opt_neighborhood(tour_edges)
        two_opt_nbh_dict = flatten_deduplicate_2opt_nbh(twoopt_nbhs, state)

        # figure out how to make repeatable sequence generation
        nbh_list = []
        num_moves_lim = 30
        reloc_nbhs_dict = list(reloc_nbhs_dict.values())
        bla = 0
        bla += len(reloc_nbhs_dict)
        if len(reloc_nbhs_dict) > num_moves_lim:
            perm = np.random.choice(len(reloc_nbhs_dict), num_moves_lim)
            reloc_nbhs_dict = [reloc_nbhs_dict[p] for p in perm]

        cross_nbh_dict = list(cross_nbh_dict.values())
        bla += len(cross_nbh_dict)
        if len(cross_nbh_dict) > num_moves_lim:
            perm = np.random.choice(len(cross_nbh_dict), num_moves_lim)
            cross_nbh_dict = [cross_nbh_dict[p] for p in perm]

        two_opt_nbh_dict = list(two_opt_nbh_dict.values())
        bla += len(two_opt_nbh_dict)
        # print(bla)
        if len(two_opt_nbh_dict) > num_moves_lim:
            perm = np.random.choice(len(two_opt_nbh_dict), num_moves_lim)
            two_opt_nbh_dict = [two_opt_nbh_dict[p] for p in perm]
        nbh_list.extend(reloc_nbhs_dict)
        nbh_list.extend(cross_nbh_dict)
        nbh_list.extend(two_opt_nbh_dict)

        return nbh_list, reloc_nbhs_dict, cross_nbh_dict, two_opt_nbh_dict

    def __init__(self, state: VRPState):
        nbh_list, relocate_nbhs, cross_nbhs, two_opt_nbhs = VRPNbH.enumerate_all_nbs(state)
        # at this point I should be passing the entire state in
        self.tour_edges, _ = enumerate_all_tours_edges(
            state.tour_idx_to_tour(), directed=True
        )
        self.tour_nodes = enumerate_all_tours_nodes(state.tour_idx_to_tour())
        self.tours = state.tours

        self.nbh_list = nbh_list

    def get_nb(self, i):
        return self.nbh_list[i]


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
        ], axis=1)  # T_src, 4
        assert np.all(src_edges_cat[:, 1] == src_nodes) and np.all(src_edges_cat[:, 2] == src_nodes)

        # relocate to depot move
        node_edges_pairs = np.concatenate([
            np.tile(src_nodes[:, None, None], (1, 1, 1)),  #  T_src, 1, 1
            np.tile(src_edges_cat[:, None, :], (1, 1, 1)),  # T_src, 1, 4
            np.tile(np.array([[[0, -1]]]), (T_src, 1, 1))  # T_src, 1, 2
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
                np.tile(src_nodes[:, None, None], (1, T_dst, 1)),  # T_src, T_dst, 1
                np.tile(src_edges_cat[:, None, :], (1, T_dst, 1)),  # T_src, T_dst, 1
                np.tile(dst_edges[None, :, :], (T_src, 1, 1))  # T_src, T_dst, 1
            ], axis=2)
            node_edges_pairs = np.reshape(node_edges_pairs, (-1, 7))

            neighborhood.append({
                'src_tour': src_tour_idx,
                'dst_tour': dst_edges_idx,
                'node_edges_pairs': node_edges_pairs
            })
    return neighborhood


def greedy_sample(
    nbhs: List[VRPNbH]
):
    actions = []
    for nbh in nbhs:
        costs = np.array([nb["cost"] for nb in nbh.nbh_list])
        action_idx = np.argmin(costs)
        actions.append(action_idx)
    # convert action indices into the action actions
    moves = [
        nbh.nbh_list[action_idx]
        for action_idx, nbh in zip(actions, nbhs)
    ]
    return actions, moves