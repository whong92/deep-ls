from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch
import copy
from deepls.gcn_model import ResidualGatedGCNModel
from deepls.gcn_layers import MLP
from deepls.VRPState import (
    VRPNbHAutoReg,
    VRPState,
    embed_cross_heuristic,
    embed_reloc_heuristic,
    get_edge_embs,
    get_node_embs,
    vectorize_reloc_moves, vectorize_cross_moves, vectorize_twopt_moves,
    normalize_vectorized_reloc_moves, normalize_vectorized_cross_moves, normalize_vectorized_twoopt_moves,
    normalize_edges,
    VRPMultiEnvAbstract, model_input_from_states
)
from typing import List, Optional, Dict, Any, Tuple
from torch.distributions import categorical as tdc
from multiprocessing import pool

from torch.optim import Adam
import numpy as np
from deepls.agent import GRCNCriticBaselineAgent
from deepls.VRPState import normalize_edges, VectorizedState


class VRPValueNet(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        config = copy.deepcopy(config)
        config['num_edge_cat_features'] = 2
        # consider consolidating with TSPGRCNValueNet since below line is the only difference
        config['node_dim'] = 3  # x, y, demand
        self.device = device
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

class VRPActionNet(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        config = copy.deepcopy(config)
        config['num_edge_cat_features'] = 2
        config['node_dim'] = 3  # x, y, demand
        self.device = device
        self.rgcn = ResidualGatedGCNModel(config)
        self.hidden_dim = config['hidden_dim']
        # TODO: feed in max demand in move mlp's
        self.first_move_mlp = MLP(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            L=3
        )

        self.cost_mlp = MLP(
            input_dim=1,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            L=1
        )
        self.cross_mlp = MLP(
            input_dim=(4 + 1) * self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            L=1
        )
        self.reloc_mlp = MLP(
            input_dim=(6 + 1) * self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            L=1
        )
        self.action_net = MLP(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            L=3
        )
        self.greedy = False
        self.pool = pool.Pool(12)

    def set_greedy(self, greedy=False):
        # set greedy decoding
        self.greedy = greedy

    def get_first_move_logits(
        self,
        x_emb,
        e_emb,
        edges_moves_vect,
        nodes_moves_vect,
    ):
        #  ############## embed and sample first move
        edges_vect_concat = []
        nodes_vect_concat = []
        num_edges = []
        num_nodes = []
        nodes_vect_all = [m['nodes'] for m in nodes_moves_vect]
        edges_vect_all = [normalize_edges(m['edge']) for m in edges_moves_vect]
        for b, (edges_vect, nodes_vect) in enumerate(zip(edges_vect_all, nodes_vect_all)):
            batch_index_edges = b * np.ones(shape=(len(edges_vect), 1))
            batch_index_nodes = b * np.ones(shape=(len(nodes_vect), 1))
            num_edges.append(len(edges_vect))
            num_nodes.append(len(nodes_vect))
            edges_vect_concat.append(np.concatenate((batch_index_edges, edges_vect), axis=1))
            nodes_vect_concat.append(np.concatenate((batch_index_nodes, nodes_vect[:, None]), axis=1))

        edges_vect_concat = torch.as_tensor(np.concatenate(edges_vect_concat, axis=0)).long()
        nodes_vect_concat = torch.as_tensor(np.concatenate(nodes_vect_concat, axis=0)).long()

        e_emb_tour = get_edge_embs(e_emb, edges_vect_concat)
        x_emb_tour = get_node_embs(x_emb, nodes_vect_concat)
        e_emb_logits = self.first_move_mlp(e_emb_tour)
        x_emb_logits = self.first_move_mlp(x_emb_tour)
        e_emb_logits = torch.split(e_emb_logits, num_edges)
        x_emb_logits = torch.split(x_emb_logits, num_nodes)
        xe_emb_logits = [
            torch.cat([x_emb_logit, e_emb_logit])
            for x_emb_logit, e_emb_logit in zip(x_emb_logits, e_emb_logits)
        ]

        xe_emb_logits_padded = pad_sequence(xe_emb_logits, batch_first=True, padding_value=-float("inf"))
        xe_emb_logits_padded = xe_emb_logits_padded.squeeze(-1)

        return xe_emb_logits_padded

    def _get_first_move_from_action_idx(self, edges_moves_vect, nodes_moves_vect, action_idxs):
        moves_0 = []
        for batch, action_idx in enumerate(action_idxs):
            edges_moves = edges_moves_vect[batch]
            nodes_moves = nodes_moves_vect[batch]
            n_node_moves = len(nodes_moves['nodes'])
            if action_idx < n_node_moves:
                moves_0.append({
                    'type': 'node',
                    'node': nodes_moves['nodes'][action_idx],
                    'tour_idx': nodes_moves['tour_idx'][action_idx]
                })
            else:
                moves_0.append({
                    'type': 'edge',
                    'edge': edges_moves['edge'][action_idx - n_node_moves],
                    'tour_idx': edges_moves['tour_idx'][action_idx - n_node_moves]
                })
        return moves_0

    def _get_second_move_from_action_idx(self, reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects, action_idxs):
        moves_1 = []
        for batch, action_idx in enumerate(action_idxs):
            _reloc_nbh_vect = reloc_nbh_vects[batch]
            _cross_nbh_vect = cross_nbh_vects[batch]
            _two_opt_nbh_vect = two_opt_nbh_vects[batch]
            n_reloc_moves = len(_reloc_nbh_vect['src_node'])
            n_cross_moves = len(_cross_nbh_vect['e0'])
            # print(n_reloc_moves, n_cross_moves, len(_two_opt_nbh_vect['e0']), action_idx)
            if action_idx < n_reloc_moves:
                move_1 = {'nb_type': 'reloc'}
                keys = ["tour0", "tour1", "src_node", "src_u", "src_v", "dst_w", "dst_wp", "src_up", "src_vp", "cost"]
                for key in keys:
                    move_1[key] = _reloc_nbh_vect[key][action_idx]
                    if len(move_1[key]) == 1: move_1[key] = move_1[key][0]
                moves_1.append(move_1)
            elif action_idx < (n_cross_moves + n_reloc_moves):
                _action_idx = action_idx - n_reloc_moves
                move_1 = {'nb_type': 'cross'}
                keys = ["tour0", "tour1", "e0", "e1", "e0p", "e1p", "cost"]
                for key in keys:
                    move_1[key] = _cross_nbh_vect[key][_action_idx]
                    if len(move_1[key]) == 1: move_1[key] = move_1[key][0]
                moves_1.append(move_1)
            else:
                _action_idx = action_idx - n_reloc_moves - n_cross_moves
                move_1 = {'nb_type': '2opt'}
                keys = ["tour_idx", "e0", "e1", "e0p", "e1p", "cost"]
                for key in keys:
                    move_1[key] = _two_opt_nbh_vect[key][_action_idx]
                    if len(move_1[key]) == 1: move_1[key] = move_1[key][0]
                moves_1.append(move_1)

        return moves_1

    @staticmethod
    def sample_moves_given_logits(
        move_logits: torch.Tensor,
        # moves: List[List[Dict[str, Any]]],
        tau: float = 1.0,
        greedy: bool = False,
        actions: Optional[torch.Tensor] = None,  # B x k
        device='cpu'
    ):
        if actions is None and not greedy:
            tour_dist_sample = tdc.Categorical(logits=move_logits / tau)
            actions = tour_dist_sample.sample()
        elif actions is None and greedy:
            actions = torch.argmax(move_logits / tau, dim=1)  # b

        tour_dist = tdc.Categorical(logits=move_logits)
        pi = tour_dist.log_prob(actions.to(device))  # b
        ent = tour_dist.entropy()
        # convert action indices into the action actions_0
        # these can be replaced by chosen moves
        # moves = [nbh[action_idx] for action_idx, nbh in zip(actions, moves)]
        return actions, pi, ent

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

    def embed_vectorized_moves(
        self,
        reloc_nbh_vects,
        cross_nbh_vects,
        twp_opt_nbh_vects,
        e_emb
    ):
        reloc_moves_embedded = embed_reloc_heuristic(
            reloc_moves_vectorized=reloc_nbh_vects,
            edge_embeddings=e_emb,
            reloc_move_mlp=self.reloc_mlp,
            cost_mlp=self.cost_mlp
        )
        cross_moves_embedded = embed_cross_heuristic(
            cross_moves_vectorized=cross_nbh_vects,
            edge_embeddings=e_emb,
            cross_move_mlp=self.cross_mlp,
            cost_mlp=self.cost_mlp
        )
        two_opt_moves_embedded = embed_cross_heuristic(
            cross_moves_vectorized=twp_opt_nbh_vects,
            edge_embeddings=e_emb,
            cross_move_mlp=self.cross_mlp,
            cost_mlp=self.cost_mlp
        )
        all_moves_embedded = [
            torch.cat([r, c, t], dim=0)
            for r, c, t in
            zip(
                reloc_moves_embedded,
                cross_moves_embedded,
                two_opt_moves_embedded
            )
        ]
        return all_moves_embedded

    def get_second_move_logits(
        self,
        e_emb,
        reloc_nbh_vects,
        cross_nbh_vects,
        two_opt_nbh_vects,
        # second_move_list
    ):
        # TODO: condition second move on first move's embedding! (but with gradients truncated)!
        all_moves_embedded = self.embed_vectorized_moves(
            reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects, e_emb
        )

        num_moves = [moves.shape[0] for moves in all_moves_embedded]
        all_moves_cat = torch.cat(all_moves_embedded, dim=0)
        all_moves_logits = self.action_net(all_moves_cat)
        all_moves_logits = torch.split(all_moves_logits, num_moves)
        all_moves_logits_padded = pad_sequence(
            all_moves_logits, batch_first=True, padding_value=-float("inf")
        ).squeeze(-1)

        return all_moves_logits_padded #, second_move_list

    def _get_first_moves_from_nbhs(self, nbhs: List[VRPNbHAutoReg]):
        edges_moves_all = []
        nodes_moves_all = []
        for nbh in nbhs:
            edges_moves_all.append(nbh.first_moves.edge_moves_vect)
            nodes_moves_all.append(nbh.first_moves.node_moves_vect)
        return edges_moves_all, nodes_moves_all

    @staticmethod
    def _get_second_moves_from_nbhs(nbhs: List[VRPNbHAutoReg]):
        reloc_nbh_vects = []
        cross_nbh_vects = []
        two_opt_nbh_vects = []
        for nbh in nbhs:
            reloc_nbh_vects.append(nbh.second_moves.reloc_nbh_vect)
            cross_nbh_vects.append(nbh.second_moves.cross_nbh_vect)
            two_opt_nbh_vects.append(nbh.second_moves.twp_opt_nbh_vect)
        return reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects


    @staticmethod
    def _make_second_move_from_state(
        state: VRPState,
        nbh: VRPNbHAutoReg,
        move_0
    ):
        return nbh._get_second_move_nbh(state, move_0)

    def _make_second_moves_multiproc(
        self,
        states: List[VRPState],
        nbhs: List[VRPNbHAutoReg],
        moves_0: List
    ):
        args = zip(states, nbhs, moves_0)
        results = self.pool.starmap(self._make_second_move_from_state, args)
        reloc_nbh_vects = []
        cross_nbh_vects = []
        two_opt_nbh_vects = []
        second_move_list = []
        for nbh, result in zip(nbhs, results):
            nbh.reloc_nbh_vect, nbh.cross_nbh_vect, nbh.twp_opt_nbh_vect, nbh.second_moves = result
            reloc_nbh_vects.append(nbh.reloc_nbh_vect)
            cross_nbh_vects.append(nbh.cross_nbh_vect)
            two_opt_nbh_vects.append(nbh.twp_opt_nbh_vect)
            second_move_list.append(nbh.second_moves)
        return reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects, second_move_list

    # append second moves from all k here
    def _make_second_moves_multiproc_env(
        self,
        moves_0: List,
        nbhs: List[VRPNbHAutoReg],
        # states: List[VRPState],
        envs: VRPMultiEnvAbstract
    ):
        reloc_nbh_vects = []
        cross_nbh_vects = []
        two_opt_nbh_vects = []
        # second_move_list = []
        # TODO: this will be run in the env
        # results = [nbh._get_second_move_nbh(state, move_0) for state, move_0, nbh in zip(states, moves_0, nbhs)]
        second_moves = envs.get_second_moves(moves_0)
        for nbh, _second_move in zip(nbhs, second_moves):
            nbh.second_moves = _second_move
            reloc_nbh_vects.append(nbh.second_moves.reloc_nbh_vect)
            cross_nbh_vects.append(nbh.second_moves.cross_nbh_vect)
            two_opt_nbh_vects.append(nbh.second_moves.twp_opt_nbh_vect)
            # second_move_list.append(nbh.second_moves)
        return reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects  # , second_move_list

    def forward_autoreg(
            self,
            # x_edges,
            x_edges_values,
            x_nodes_coord,
            x_tour,
            x_best_tour,
            # states: List[VRPState],
            envs: VRPMultiEnvAbstract
    ):
        x_cat = torch.stack([x_tour, x_best_tour], dim=3)
        x_emb, e_emb = self.rgcn(x_cat, x_edges_values, x_nodes_coord)
        # x_emb = self.rgcn(x_nodes_coord)

        # TODO instantiate without state, get first moves from env
        # nbhs = [VRPNbHAutoReg.init_from_state(state) for state in states]
        first_moves = envs.get_first_moves()
        nbhs = [VRPNbHAutoReg(first_moves=_first_moves) for _first_moves in first_moves]
        edges_moves_all, nodes_moves_all = self._get_first_moves_from_nbhs(nbhs)
        first_move_logits = self.get_first_move_logits(
            x_emb,
            e_emb,
            edges_moves_all, nodes_moves_all
        )
        actions_0, pi_0, _ = self.sample_moves_given_logits(first_move_logits, device=self.device, greedy=False)
        moves_0 = self._get_first_move_from_action_idx(edges_moves_all, nodes_moves_all, actions_0)
        reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects = \
            self._make_second_moves_multiproc_env(moves_0, nbhs, envs)

        reloc_nbh_vects_normalized, cross_nbh_vects_normalized, two_opt_nbh_vects_normalized = [], [], []
        for _reloc_nbh_vects, _cross_nbh_vects, _two_opt_nbh_vects in zip(
            reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects
        ):
            reloc_nbh_vects_normalized.append(normalize_vectorized_reloc_moves(_reloc_nbh_vects))
            cross_nbh_vects_normalized.append(normalize_vectorized_cross_moves(_cross_nbh_vects))
            two_opt_nbh_vects_normalized.append(normalize_vectorized_twoopt_moves(_two_opt_nbh_vects))

        # assert False
        second_moves_logits_padded = \
            self.get_second_move_logits(e_emb, reloc_nbh_vects_normalized, cross_nbh_vects_normalized,
                                        two_opt_nbh_vects_normalized)
        actions_1, pi_1, _ = self.sample_moves_given_logits(second_moves_logits_padded,
                                                            device=self.device, greedy=False)
        moves_1 = self._get_second_move_from_action_idx(reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects, actions_1)
        actions = torch.stack([actions_0, actions_1], dim=1)
        pi = pi_0 + pi_1
        moves = moves_1
        return moves, pi, actions, nbhs

    def forward(
        self,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        env
    ):
        return self.forward_autoreg(x_edges_values, x_nodes_coord, x_tour, x_best_tour, env)

    def get_action_pref_autoreg(
        self,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        actions: torch.Tensor,
        nbhs: List[VRPNbHAutoReg]
    ):
        """
        """
        x_cat = torch.stack([x_tour, x_best_tour], dim=3)
        x_emb, e_emb = self.rgcn(x_cat, x_edges_values, x_nodes_coord)

        edges_moves_all, nodes_moves_all = self._get_first_moves_from_nbhs(nbhs)
        first_move_logits = self.get_first_move_logits(
            x_emb,
            e_emb,
            edges_moves_all, nodes_moves_all
        )

        actions_0, pi_0, ent_0 = self.sample_moves_given_logits(first_move_logits, device=self.device, greedy=False, actions=actions[:, 0])

        reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects = (
            self._get_second_moves_from_nbhs(nbhs)
        )

        # move the following into get_second_move_logits?
        reloc_nbh_vects_normalized, cross_nbh_vects_normalized, two_opt_nbh_vects_normalized = [], [], []
        for _reloc_nbh_vects, _cross_nbh_vects, _two_opt_nbh_vects in zip(
                reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects
        ):
            reloc_nbh_vects_normalized.append(normalize_vectorized_reloc_moves(_reloc_nbh_vects))
            cross_nbh_vects_normalized.append(normalize_vectorized_cross_moves(_cross_nbh_vects))
            two_opt_nbh_vects_normalized.append(normalize_vectorized_twoopt_moves(_two_opt_nbh_vects))

        # assert False
        second_moves_logits_padded = \
            self.get_second_move_logits(e_emb, reloc_nbh_vects_normalized, cross_nbh_vects_normalized,
                                        two_opt_nbh_vects_normalized)
        actions_1, pi_1, ent_1 = self.sample_moves_given_logits(second_moves_logits_padded, device=self.device, greedy=False, actions=actions[:, 1])
        pi = pi_0 + pi_1
        ent = ent_0 + ent_1
        moves = None

        return moves, pi, actions, ent

    def get_action_pref(
            self,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        actions: torch.Tensor,
        nbhs: List[VRPNbHAutoReg]
    ):
        return self.get_action_pref_autoreg(
            # x_edges,
            x_edges_values,
            x_nodes_coord,
            x_tour,
            x_best_tour,
            # states,
            actions,
            nbhs
        )


def concat_states(vectorized_states: List[VectorizedState]):
    vs = vectorized_states
    return VectorizedState(
        x_tour=np.concatenate([s.x_tour for s in vs], axis=0),
        x_best_tour=np.concatenate([s.x_best_tour for s in vs], axis=0),
        x_nodes_coord=np.concatenate([s.x_nodes_coord for s in vs], axis=0),
        # x_edges=np.concatenate([s.x_edges for s in vs], axis=0),
        x_edges_values=np.concatenate([s.x_edges_values for s in vs], axis=0),
        state_ids=np.concatenate([s.state_ids for s in vs], axis=0),
        states_cost=np.concatenate([s.states_cost for s in vs], axis=0),
        best_states_cost=np.concatenate([s.best_states_cost for s in vs], axis=0),
        states_opt_cost=np.concatenate([s.states_opt_cost for s in vs], axis=0),
    )


class ActionNetRunner:
    """
    wraps a bunch of methods that can be re-used to run the policy
    """

    def __init__(self, net: VRPActionNet, device):
        self.net = net
        self.device = device

    def policy(self, vectorized_states: List[VectorizedState], env):
        """
        :param states: sequence of tuple of 2 TSP2OptEnv states - the current state and the best state so far
        :return:
            actions - the actions sampled for this set of states
            cache - stuff that the policy net expects to be cached (to avoid re-computation), and returned to it in
            the list of experiences which is given in e.g. get_action_pref method
        """
        # best_states = [state[1] for state in states]
        # states: List[VRPState] = [state[0] for state in states]
        # # cur state
        # x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour, states_input = \
        #     list(model_input_from_states(states, best_states))

        # B = len(vectorized_states)
        vectorized_state = concat_states(vectorized_states)

        model_input = [
            # torch.as_tensor(np.ones_like(vectorized_state.x_edges_values)),
            torch.as_tensor(vectorized_state.x_edges_values),
            torch.as_tensor(vectorized_state.x_nodes_coord),
            torch.as_tensor(vectorized_state.x_tour),
            torch.as_tensor(vectorized_state.x_best_tour)
        ]
        with torch.no_grad():
            # moves, pis, action_idxs, nbhs = self.net(*[t.clone().to(self.device) for t in model_input] + [states_input], env)
            moves, pis, action_idxs, nbhs = self.net(*[t.clone().to(self.device) for t in model_input],
                                                     env)

        cache = {
            'model_input': model_input,
            'action': action_idxs.detach().to('cpu'),
            'action_pref': pis.detach().to('cpu'),
            # 'tour_len': [state.get_cost() for state in states],
            'moves': moves,
            # 'state_ids': [state.id for state in states],
            'state_ids': vectorized_state.state_ids,  # [state.id for state in states],
            'nbhs': nbhs  # maybe construct the nbh in agent so agent owns it?
        }
        actions = [{'move': move, 'terminate': False} for move in moves]
        return actions, cache

    def get_action_pref(self, experiences, perm):
        """
        given sampled experiences,
        :param experiences:
        :param perm:
        :return:
        action pref, h_sa - may contain gradient
        """
        cached_inputs = [e['cache']['model_input'] for e in experiences]
        x_edges_values = torch.cat([c[0] for c in cached_inputs], dim=0)[perm]
        x_nodes_coord = torch.cat([c[1] for c in cached_inputs], dim=0)[perm]
        x_tour = torch.cat([c[2] for c in cached_inputs], dim=0)[perm]
        x_best_tour = torch.cat([c[3] for c in cached_inputs], dim=0)[perm]

        nbhs = []
        for e in experiences:
            nbhs.extend(e['cache']['nbhs'])
        nbhs = [nbhs[p] for p in perm]

        actions = torch.cat([e['cache']['action'] for e in experiences], dim=0)[perm]
        h_sa_old = torch.cat([e['cache']['action_pref'] for e in experiences], dim=0)[perm]

        model_inputs = [x_edges_values, x_nodes_coord, x_tour, x_best_tour]

        _, h_sa, _, ent = self.net.get_action_pref(
            *[t.clone().to(self.device) for t in model_inputs] +
             # [states_input] +
             [actions.clone()],
            nbhs=nbhs
        )
        return h_sa, h_sa_old.to(self.device), ent

from deepls.agent import BaseAgent, ExperienceBuffer, iter_over_tensor_minibatch, PPO_EPS
import pandas as pd

class AverageStateRewardBaselineAgentVRP(BaseAgent):

    def init_replay_buffer(self, replay_buffer_size):
        # initialize replay buffer
        self.replay_buffer = ExperienceBuffer(replay_buffer_size, copy=False)

    def agent_init(self, agent_config={}):
        """Setup variables to track state
        """
        self.last_state = None
        self.last_action = None
        self.last_cache = None  # anything that needs caching for timestep t-1

        self.episode_steps = 0
        self.episode = -1

        self.train = True  # whether or not to perform optimization

        self._agent_init(agent_config)
        self.init_replay_buffer(agent_config['replay_buffer_size'])
        self.batch_size = agent_config['batch_sz']
        self.gamma = agent_config.get('gamma', 1.0)

        self.actions = []
        self.caches = []
        self.rewards = []


    def agent_start(self, state, env=None):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Sequence[TSP2OptState]): the (multi)state from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.episode_steps = 0
        self.episode += 1

        self.state_ids = []
        self.actions = []
        self.caches = []
        self.rewards = []

        self.last_state = state
        self.state_ids.append([s.state_ids[0] for s in self.last_state])
        self.last_action, self.last_cache = self.policy(self.last_state, env)
        self.actions.append(self.last_action)
        self.caches.append(self.last_cache)
        return self.last_action


    # weights update using optimize_network, and updating last_state and last_action (~5 lines).
    def agent_step(self, reward, state, env=None):
        """A step taken by the agent.
        Args:
            reward (Sequence[float]): the rewards received for taking the last action taken
            state (Sequence[TSP2OptState]): the (multi)state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        self.episode_steps += 1
        self.rewards.append(reward)
        # Select action
        action, cache = self.policy(state, env=env)

        # Update the last state and last action.
        self.last_state = state
        self.last_action = action
        self.last_cache = cache

        # self.state_ids.append([s[0].id for s in self.last_state])
        self.state_ids.append([s.state_ids[0] for s in self.last_state])
        self.actions.append(self.last_action)
        self.caches.append(self.last_cache)

        return action

    def set_train(self):
        self.train = True

    def set_eval(self):
        self.train = False

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (Sequence[float]): the rewards the agent received for entering the
                terminal state.
        """
        self.episode_steps += 1

        if self.train:
            for s, (_state_id, _next_state, _action, _reward, _cache) in enumerate(
                zip(self.state_ids[:-1], self.state_ids[1:], self.actions, self.rewards, self.caches), start=1
            ):
                self.replay_buffer.append(
                    self.episode,
                    _state_id,
                    _action,
                    _reward,
                    False,
                    None, # _next_state,
                    s,
                    _cache
                )
            # Append new experience to replay buffer
            self.replay_buffer.append(
                self.episode,
                # [s[0].id for s in self.last_state],
                self.state_ids.append([s.state_ids[0] for s in self.last_state]),
                self.last_action,
                reward,
                True,
                None,
                self.episode_steps,
                self.last_cache
            )

            # compute returns - no discounting for now
            experience = self.replay_buffer.get_episode(self.episode)
            # building the discount matrix
            # B x n_steps
            reward = torch.as_tensor([step['reward'] for step in experience]).T

            n_steps = reward.shape[1]
            gamma = self.gamma
            d_r0 = torch.pow(gamma, torch.arange(0, n_steps))
            d_mat = torch.zeros(size=(n_steps, n_steps))

            for r in range(n_steps):
                d_mat[r, r:] = d_r0[:n_steps - r]
            # B x n_steps x n_steps (to sum over)
            rewards_mat = reward[:, None, :] * d_mat[None, :, :]
            returns = torch.sum(rewards_mat, dim=2).T  # n_steps x B

            for ret, step in zip(returns, experience):
                state_ids = step['cache']['state_ids']
                df = pd.DataFrame({'state_ids': state_ids, 'returns': ret.numpy()})
                avg_ret = np.array(df.groupby('state_ids', as_index=False).returns.transform(np.mean).returns)
                step['cache']['return'] = ret
                step['cache']['average_return'] = torch.as_tensor(avg_ret)
            # Perform replay steps:
            experiences_dict = self.replay_buffer.sample_experience(self.replay_buffer.get_size())
            return self.agent_optimize(experiences_dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _agent_init(self, agent_config):
        self.agent_config = copy.deepcopy(agent_config)
        model_config = agent_config['model']
        optimizer_config = agent_config['optim']
        device = agent_config.get('device', 'cpu')
        self.device = device

        self.net = VRPActionNet(model_config, device).to(device)
        self.optimizer = Adam(
            self.net.parameters(),
            lr=optimizer_config['step_size'],
            betas=(optimizer_config['beta_m'], optimizer_config['beta_v']),
            eps=optimizer_config['epsilon']
        )
        self.action_net_runner = ActionNetRunner(self.net, device)
        self.dont_optimize_policy_steps = model_config.get('dont_optimize_policy_steps', 0)
        self.critic_abs_err = None

        self.policy_optimize_every = agent_config['policy_optimize_every']
        self.minibatch_size = agent_config['minibatch_sz']
        self.use_ppo_update = agent_config.get('use_ppo_update', False)
        self.entropy_bonus = agent_config.get('entropy_bonus', 0.)
        self.greedy = False

    def compute_baseline(self, experiences, perm):
        average_returns = torch.cat([e['cache']['average_return'] for e in experiences], dim=0)
        average_returns = average_returns[perm]
        return average_returns

    def save(self, path):
        bla = {
            'agent_config': self.agent_config,
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(bla, path)
        
    def load(self, path, init_config=True, load_optim=True, device=None):
        if device is not None:
            self.device = device
        bla = torch.load(path, map_location=self.device)
        agent_config = bla['agent_config']
        agent_config['device'] = self.device
        if init_config:
            self.agent_init(agent_config)
        self.net.load_state_dict(bla['net'], strict=False)
        if load_optim:
            self.optimizer.load_state_dict(bla['optimizer'])

    def agent_optimize(self, experiences):
        """
        run this with provided experiences to run one step of optimization
        """
        optimize_policy = (self.episode % self.policy_optimize_every == 0) \
                          and (self.episode > self.dont_optimize_policy_steps)

        # always try and get an estimate of the critic abs_err
        returns = torch.cat([e['cache']['return'] for e in experiences], dim=0).float()
        if len(returns) < self.batch_size:
            return
        perm = np.random.choice(len(returns), self.batch_size, replace=False)
        returns = returns[perm]
        baseline = self.compute_baseline(experiences, perm)
        self.critic_abs_err = torch.abs(baseline - returns).mean().detach().to('cpu')

        if optimize_policy:
            # metrics to log
            policy_loss_accum = 0.
            td_err_accum = 0.
            ratio_accum = 0.
            elig_loss_accum = 0.
            policy_entropy_accum = 0.

            self.optimizer.zero_grad()
            for (minibatch_perm, minibatch_returns, minibatch_baseline), weight in \
                    iter_over_tensor_minibatch(perm, returns, baseline, minibatch_size=self.minibatch_size):
                action_pref_output = self.action_net_runner.get_action_pref(experiences, minibatch_perm)
                td_err = (minibatch_returns.clone() - minibatch_baseline.detach()).to(self.device)

                h_sa = action_pref_output[0]
                h_sa_old = action_pref_output[1]
                ratio = torch.exp(h_sa - h_sa_old)
                ratio_accum += torch.sum(ratio.cpu().detach())

                if len(action_pref_output) > 2:
                    policy_entropy = action_pref_output[2]
                    policy_entropy_accum += torch.sum(policy_entropy.cpu().detach())
                else:
                    policy_entropy = 0.
                if self.use_ppo_update:
                    # PPO update
                    policy_loss_p1 = -td_err * ratio
                    policy_loss_p2 = -td_err * torch.clip(ratio, 1. - PPO_EPS, 1. + PPO_EPS)
                    policy_loss = torch.maximum(policy_loss_p1, policy_loss_p2).mean() * weight
                else:
                    # regular PG update
                    eligibility_loss = -h_sa  # NLL loss
                    # entropy bonus
                    beta = self.entropy_bonus
                    # TODO: discounting!(?)
                    policy_loss = (td_err * eligibility_loss - beta * policy_entropy).mean() * weight

                policy_loss_accum += policy_loss.cpu().detach()
                elig_loss_accum += torch.sum(h_sa.cpu().detach())
                td_err_accum += torch.sum(td_err.cpu().detach())

                policy_loss.backward()

            metrics = {
                'policy_loss_accum': float(policy_loss_accum),
                'td_err_accum': float(td_err_accum / self.batch_size),
                'ratio_accum': float(ratio_accum / self.batch_size),
                'elig_loss_accum': float(elig_loss_accum / self.batch_size),
                'policy_entropy_accum': float(policy_entropy_accum / self.batch_size),
            }

            self.optimizer.step()
            return metrics

    def set_greedy(self, greedy=False):
        self.greedy = greedy
        self.net.set_greedy(greedy)

    def policy(self, states, env=None):
        """
        run this with provided state to get action
        """
        return self.action_net_runner.policy(states, env)


class CriticBaselineAgentVRP(GRCNCriticBaselineAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _agent_init(self, agent_config):
        self.agent_config = copy.deepcopy(agent_config)
        model_config = agent_config['model']
        optimizer_config = agent_config['optim']
        device = agent_config.get('device', 'cpu')
        self.device = device

        self.net = VRPActionNet(model_config, device).to(device)
        self.optimizer = Adam(
            self.net.parameters(),
            lr=optimizer_config['step_size'],
            betas=(optimizer_config['beta_m'], optimizer_config['beta_v']),
            eps=optimizer_config['epsilon']
        )
        self.action_net_runner = ActionNetRunner(self.net, device)

        self.value_net_type = model_config.get('value_net_type', 'normal')
        if self.value_net_type == 'normal':
            self.critic_baseline = VRPValueNet(model_config, device).to(self.device)
            self.critic_loss = torch.nn.HuberLoss(delta=0.2).to(self.device)
        elif self.value_net_type == 'lognormal':
            raise ValueError("LogNormal Value Net not supported")
        else:
            raise ValueError(f"value_net_type not recognized: {self.value_net_type}")

        self.critic_optimizer = Adam(
            self.critic_baseline.parameters(),
            lr=optimizer_config['step_size_critic'],
            betas=(optimizer_config['beta_m'], optimizer_config['beta_v']),
            eps=optimizer_config['epsilon']
        )
        self.dont_optimize_policy_steps = model_config.get('dont_optimize_policy_steps', 0)
        self.critic_loss_val = None
        self.critic_abs_err = None

        self.policy_optimize_every = agent_config['policy_optimize_every']
        self.critic_optimize_every = agent_config['critic_optimize_every']
        self.minibatch_size = agent_config['minibatch_sz']
        self.use_ppo_update = agent_config.get('use_ppo_update', False)
        self.entropy_bonus = agent_config.get('entropy_bonus', 0.)
        self.greedy = False


VRP_STANDARD_PROBLEM_CONF = {
    10: {
        'size': 10,
        'capacity': 4,
    },
    20: {
        'size': 20,
        'capacity': 6,
    },
    50: {
        'size': 50,
        'capacity': 8,
    },
    100: {
        'size': 100,
        'capacity': 10,
    },
}


if __name__ == "__main__":

    episodes = 10000
    N = 20
    num_steps = 20
    max_tour_demand = VRP_STANDARD_PROBLEM_CONF[N]['capacity']

    agent_config = {
        'replay_buffer_size': 10,
        'batch_sz': 64,
        'minibatch_sz': 32,
        'policy_optimize_every': 2,
        'critic_optimize_every': 1,
        'dont_optimize_policy_steps': 1000,
        'model': {
            "node_dim": 2,
            "voc_edges_in": 3,
            "hidden_dim": 128,
            "num_layers": 3,
            "mlp_layers": 3,
            "aggregation": "mean",
            "num_edge_cat_features": 2
        },
        'optim': {
            'step_size': 1e-4,
            'step_size_critic': 2e-4,
            'beta_m': 0.9,
            'beta_v': 0.999,
            'epsilon': 1e-8
        },
        'device': 'cuda'
    }

    agent = AverageStateRewardBaselineAgentVRP()
    agent.agent_init(agent_config)
    # agent.load('model-020-nodes-040-h-032-steps-small-lr-final.ckpt', init_config=False)
    moving_avg = None

    from tqdm import tqdm
    from deepls.VRPState import VRPMultiRandomEnv
    moving_avgs = []

    # env = VRPEnvRandom(num_nodes=N, max_num_steps=num_steps, max_tour_demand=max_tour_demand)
    envs = VRPMultiRandomEnv(
        num_nodes=N,
        max_num_steps=num_steps,
        max_tour_demand=max_tour_demand,
        num_samples_per_instance=12,
        num_instance_per_batch=1
    )
    pbar = tqdm(range(episodes))

    for episode in pbar:

        # env.set_instance_as_state(instance, id=episode, max_num_steps=num_steps)
        states = envs.reset(fetch_next=True)
        actions = agent.agent_start(states)
        init_cost = states[0][1].get_cost(exclude_depot=False)
        while True:
            states, rewards, dones = envs.step(actions)
            done = dones[0]
            if done:
                if moving_avg is None:
                    moving_avg = np.mean(rewards)
                else:
                    moving_avg = 0.9 * moving_avg + 0.1 * np.mean(rewards)

                agent.moving_avg = moving_avg
                # print(rewards)
                agent.agent_end(np.array(rewards))
                break
            else:
                actions = agent.agent_step(
                    np.array(rewards),
                    states
                )
        moving_avgs.append(moving_avg)
        best_state_cost = states[0][1].get_cost(exclude_depot=False)
        desc = f"best_state cost: {best_state_cost:.3f} / {init_cost:.3f} = {best_state_cost / init_cost:.3f}, rew ma: {moving_avg:.3f}"
        pbar.set_description(desc)

    hidden_dim = agent_config['model']['hidden_dim']
    agent.save(
        f'model-{N:03d}-nodes-{num_steps:03d}-h-{hidden_dim:03d}-steps-small-lr-final-add-ent-bonus.ckpt'
    )

    import matplotlib.pyplot as plt
    plt.plot(moving_avgs)
    plt.show()