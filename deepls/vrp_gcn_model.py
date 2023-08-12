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
    vectorize_reloc_moves, vectorize_cross_moves, vectorize_twopt_moves
)
from typing import List, Optional, Dict, Any, Tuple
from torch.distributions import categorical as tdc
from multiprocessing import pool

from torch.optim import Adam
import numpy as np
from deepls.agent import AverageStateRewardBaselineAgent

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

        self.cross_mlp = MLP(
            input_dim=4 * self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            L=1
        )
        self.reloc_mlp = MLP(
            input_dim=6 * self.hidden_dim,
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
        edges_vect_all, nodes_vect_all, first_move_nbhs
    ):
        #  ############## embed and sample first move
        edges_vect_concat = []
        nodes_vect_concat = []
        num_edges = []
        num_nodes = []
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

        return xe_emb_logits_padded, first_move_nbhs

    @staticmethod
    def sample_moves_given_logits(
        move_logits: torch.Tensor,
        moves: List[List[Dict[str, Any]]],
        tau: float = 1.0,
        greedy: bool = False,
        actions: Optional[torch.Tensor] = None,
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
        moves = [nbh[action_idx] for action_idx, nbh in zip(actions, moves)]
        return actions, moves, pi, ent

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
            reloc_move_mlp=self.reloc_mlp
        )
        cross_moves_embedded = embed_cross_heuristic(
            cross_moves_vectorized=cross_nbh_vects,
            edge_embeddings=e_emb,
            cross_move_mlp=self.cross_mlp
        )
        two_opt_moves_embedded = embed_cross_heuristic(
            cross_moves_vectorized=twp_opt_nbh_vects,
            edge_embeddings=e_emb,
            cross_move_mlp=self.cross_mlp
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
        reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects, second_move_list
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

        return all_moves_logits_padded, second_move_list

    def _get_first_moves_from_nbhs(self, nbhs: List[VRPNbHAutoReg]):
        edges_vect_all = []
        nodes_vect_all = []
        first_move_nbhs = []
        for nbh in nbhs:
            first_move_nbhs.append(nbh.first_move_nbh)
            edges_vect_all.append(nbh.edges_vect)
            nodes_vect_all.append(nbh.nodes_vect)
        return edges_vect_all, nodes_vect_all, first_move_nbhs

    @staticmethod
    def _get_second_moves_from_nbhs(nbhs: List[VRPNbHAutoReg]):
        reloc_nbh_vects = []
        cross_nbh_vects = []
        two_opt_nbh_vects = []
        second_move_list = []
        for nbh in nbhs:
            reloc_nbh_vects.append(nbh.reloc_nbh_vect)
            cross_nbh_vects.append(nbh.cross_nbh_vect)
            two_opt_nbh_vects.append(nbh.twp_opt_nbh_vect)
            second_move_list.append(nbh.second_moves)
        return reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects, second_move_list

    @staticmethod
    def _make_second_moves_from_states(
        states: List[VRPState],
        nbhs: List[VRPNbHAutoReg],
        moves_0: List
    ):
        reloc_nbh_vects = []
        cross_nbh_vects = []
        two_opt_nbh_vects = []
        second_move_list = []
        for state, nbh, move_0 in zip(states, nbhs, moves_0):
            nbh.reloc_nbh_vect, nbh.cross_nbh_vect, nbh.twp_opt_nbh_vect, nbh.second_moves = \
                nbh._get_second_move_nbh(state, move_0)
            reloc_nbh_vects.append(nbh.reloc_nbh_vect)
            cross_nbh_vects.append(nbh.cross_nbh_vect)
            two_opt_nbh_vects.append(nbh.twp_opt_nbh_vect)
            second_move_list.append(nbh.second_moves)
        return reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects, second_move_list

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

    def forward_autoreg(
        self,
        x_edges,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        states: List[VRPState],
    ):
        x_cat = torch.stack([x_tour, x_best_tour], dim=3)
        x_emb, e_emb = self.rgcn(x_cat, x_edges_values, x_nodes_coord)

        # TODO: feed this into move mlps, actually maybe no need? since it's always 1 anyways
        # ;max_demands = torch.as_tensor([state.max_tour_demand for state in states]).to(self.device)
        # copy the nbhs so we can modify it
        nbhs = [copy.deepcopy(state.get_nbh()) for state in states]

        edges_vect_all, nodes_vect_all, first_move_nbhs = self._get_first_moves_from_nbhs(nbhs)
        first_move_logits, first_move_nbhs = self.get_first_move_logits(
            x_emb,
            e_emb,
            edges_vect_all, nodes_vect_all, first_move_nbhs
        )
        actions_0, moves_0, pi_0, _ = self.sample_moves_given_logits(first_move_logits, first_move_nbhs, device=self.device)
        # print(moves_0)
        # reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects, second_move_list = self._get_second_move_nbh(
        #     states, moves_0
        # )
        # TODO: use process pool to parallelize this maybe?
        # reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects, second_move_list = \
        #     self._make_second_moves_from_states(states, nbhs, moves_0)
        reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects, second_move_list = \
            self._make_second_moves_multiproc(states, nbhs, moves_0)

        # assert False
        second_moves_logits_padded, second_move_list = \
            self.get_second_move_logits(e_emb, reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects, second_move_list)
        actions_1, moves_1, pi_1, _ = self.sample_moves_given_logits(second_moves_logits_padded, second_move_list, device=self.device)
        # print(moves_1)
        #
        actions = torch.stack([actions_0, actions_1], dim=1)
        pi = pi_0 + pi_1
        moves = moves_1

        return moves, pi, actions, nbhs

    def embed_state_and_neighborhood(
        self,
        x_edges,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        states: List[VRPState],
    ) -> List[torch.Tensor]:
        x_cat = torch.stack([x_tour, x_best_tour], dim=3)
        nbhs = [state.get_nbh() for state in states]

        # action 0
        x_emb, e_emb = self.rgcn(x_cat, x_edges_values, x_nodes_coord)

        reloc_moves_embedded = embed_reloc_heuristic(
            reloc_moves_vectorized=[nbh.relocate_nbhs_vectorized for nbh in nbhs],
            edge_embeddings=e_emb,
            reloc_move_mlp=self.reloc_mlp
        )
        cross_moves_embedded = embed_cross_heuristic(
            cross_moves_vectorized=[nbh.cross_nbhs_vectorized for nbh in nbhs],
            edge_embeddings=e_emb,
            cross_move_mlp=self.cross_mlp
        )
        two_opt_moves_embedded = embed_cross_heuristic(
            cross_moves_vectorized=[nbh.two_opt_nbhs_vectorized for nbh in nbhs],
            edge_embeddings=e_emb,
            cross_move_mlp=self.cross_mlp
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

    def forward(
        self,
        x_edges,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        states: List[VRPState]
    ):
        return self.forward_autoreg(x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour, states)

    def get_action_pref_autoreg(
        self,
        x_edges,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        states: List[VRPState],
        actions: torch.Tensor,
        nbhs: List[VRPNbHAutoReg]
    ):
        """
        """
        x_cat = torch.stack([x_tour, x_best_tour], dim=3)
        x_emb, e_emb = self.rgcn(x_cat, x_edges_values, x_nodes_coord)

        # TODO: feed this into move mlps
        max_demands = torch.as_tensor([state.max_tour_demand for state in states]).to(self.device)

        edges_vect_all, nodes_vect_all, first_move_nbhs = self._get_first_moves_from_nbhs(nbhs)
        first_move_logits, first_move_nbhs = self.get_first_move_logits(
            x_emb,
            e_emb,
            edges_vect_all, nodes_vect_all, first_move_nbhs
        )
        actions_0, moves_0, pi_0, ent_0 = self.sample_moves_given_logits(first_move_logits, first_move_nbhs, actions=actions[:, 0], device=self.device)

        reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects, second_move_list = self._get_second_moves_from_nbhs(nbhs)
        second_moves_logits_padded, second_move_list = self.get_second_move_logits(
            e_emb, reloc_nbh_vects, cross_nbh_vects, two_opt_nbh_vects, second_move_list
        )
        actions_1, moves_1, pi_1, ent_1 = self.sample_moves_given_logits(second_moves_logits_padded, second_move_list, actions=actions[:, 1], device=self.device)

        pi = pi_0 + pi_1
        ent = ent_0 + ent_1
        moves = list(zip(moves_0, moves_1))

        return moves, pi, actions, ent

    def get_action_pref(
        self,
        x_edges,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        states: List[VRPState],
        actions: torch.Tensor,
        nbhs: List[VRPNbHAutoReg]
    ):
        return self.get_action_pref_autoreg(
            x_edges,
            x_edges_values,
            x_nodes_coord,
            x_tour,
            x_best_tour,
            states,
            actions,
            nbhs
        )

def model_input_from_states(states: List[VRPState], best_states: List[VRPState]):
    x_edges = []
    x_edges_values = []
    x_nodes_coord = []
    x_tour = []
    x_best_tour = []
    states_input = []
    for state, best_state in zip(states, best_states):
        x_tour.append(
            torch.Tensor(
                state.get_tours_adj(directed=False, sum=True)
            ).unsqueeze(0).to(torch.long)
        )
        x_best_tour.append(
            torch.Tensor(
                best_state.get_tours_adj(directed=False, sum=True)
            ).unsqueeze(0).to(torch.long)
        )
        x_edges.append(torch.ones_like(x_tour[-1]))
        x_edges_values.append(torch.Tensor(state.edge_weights).unsqueeze(0))

        # add batch dimension and feature dimension
        node_demands = state.get_node_demands(include_depot=True)[None, :, None]
        node_coords = state.nodes_coord[None, :]

        x_nodes_coord.append(
            torch.as_tensor(np.concatenate((node_coords, node_demands), axis=2)).to(torch.float)
        )
        states_input.append(copy.deepcopy(state))
    return (
        torch.cat(x_edges, dim=0),
        torch.cat(x_edges_values, dim=0),
        torch.cat(x_nodes_coord, dim=0),
        torch.cat(x_tour, dim=0),
        torch.cat(x_best_tour, dim=0),
        states_input
    )


class ActionNetRunner:
    """
    wraps a bunch of methods that can be re-used to run the policy
    """
    def __init__(self, net: VRPActionNet, device):
        self.net = net
        self.device = device

    def policy(self, states: List[Tuple[VRPState, VRPState]]):
        """
        :param states: sequence of tuple of 2 TSP2OptEnv states - the current state and the best state so far
        :return:
            actions - the actions sampled for this set of states
            cache - stuff that the policy net expects to be cached (to avoid re-computation), and returned to it in
            the list of experiences which is given in e.g. get_action_pref method
        """
        best_states = [state[1] for state in states]
        states: List[VRPState] = [state[0] for state in states]
        # cur state
        x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour, states_input = \
            list(model_input_from_states(states, best_states))

        model_input = [x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour]
        with torch.no_grad():
            moves, pis, action_idxs, nbhs = self.net(*[t.clone().to(self.device) for t in model_input] + [states_input])
        cache = {
            'model_input': model_input + [states_input],
            'action': action_idxs.detach().to('cpu'),
            'action_pref': pis.detach().to('cpu'),
            'tour_len': [state.get_cost() for state in states],
            'moves': moves,
            'nbhs': nbhs
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
        x_edges = torch.cat([c[0] for c in cached_inputs], dim=0)[perm]
        x_edges_values = torch.cat([c[1] for c in cached_inputs], dim=0)[perm]
        x_nodes_coord = torch.cat([c[2] for c in cached_inputs], dim=0)[perm]
        x_tour = torch.cat([c[3] for c in cached_inputs], dim=0)[perm]
        x_best_tour = torch.cat([c[4] for c in cached_inputs], dim=0)[perm]
        states_input = []

        for c in cached_inputs:
            states_input.extend(c[5])
        states_input = [states_input[p] for p in perm]

        nbhs = []
        for e in experiences:
            nbhs.extend(e['cache']['nbhs'])
        nbhs = [nbhs[p] for p in perm]

        actions = torch.cat([e['cache']['action'] for e in experiences], dim=0)[perm]
        h_sa_old = torch.cat([e['cache']['action_pref'] for e in experiences], dim=0)[perm]

        model_inputs = [x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour]

        _, h_sa, _, ent = self.net.get_action_pref(
            *[t.clone().to(self.device) for t in model_inputs] +
             [states_input] +
             [actions.clone()],
            nbhs=nbhs
        )
        return h_sa, h_sa_old.to(self.device), ent


class AverageStateRewardBaselineAgentVRP(AverageStateRewardBaselineAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moving_avg = None

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