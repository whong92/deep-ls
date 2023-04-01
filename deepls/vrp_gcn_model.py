import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch
import copy
from deepls.gcn_model import ResidualGatedGCNModel
from deepls.gcn_layers import MLP
from deepls.VRPState import VRPNbH, embed_cross_heuristic, embed_reloc_heuristic
from typing import List, Optional
from torch.distributions import categorical as tdc


class VRPActionNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = copy.deepcopy(config)
        config['num_edge_cat_features'] = 2
        self.rgcn = ResidualGatedGCNModel(config)
        self.hidden_dim = config['hidden_dim']
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

    def set_greedy(self, greedy=False):
        # set greedy decoding
        self.greedy = greedy

    @staticmethod
    def _sample_actions(
        nbh_logits: torch.Tensor,
        nbhs: List[VRPNbH],
        tau: float = 1.0,
        greedy: bool = False,
        actions: Optional[torch.Tensor] = None,
    ):
        batch_size = nbh_logits.shape[0]
        if actions is None and not greedy:
            tour_dist_sample = tdc.Categorical(logits=nbh_logits / tau)
            actions = tour_dist_sample.sample()
        elif actions is None and greedy:
            actions = torch.argmax(nbh_logits / tau, dim=1)  # b
        tour_dist = tdc.Categorical(logits=nbh_logits / tau)
        # calculate log_probs for action
        pi = tour_dist.log_prob(actions)  # b
        # convert action indices into the action actions
        moves = [nbh.nbh_list[action_idx] for action_idx, nbh in zip(actions, nbhs)]
        return actions, moves, pi

    def get_nbh_logits(
        self,
        x_edges,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        nbhs: List[VRPNbH],
        pad: bool = True
    ):
        all_moves_embedded = self.embed_state_and_neighborhood(
            x_edges,
            x_edges_values,
            x_nodes_coord,
            x_tour,
            x_best_tour,
            nbhs
        )
        num_moves = [moves.shape[0] for moves in all_moves_embedded]
        all_moves_cat = torch.cat(all_moves_embedded, dim=0)
        all_moves_logits = self.action_net(all_moves_cat)
        all_moves_logits = torch.split(all_moves_logits, num_moves)
        if pad:
            return pad_sequence(all_moves_logits, batch_first=True, padding_value=-float("inf"))
        return all_moves_logits

    def embed_state_and_neighborhood(
        self,
        x_edges,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        nbhs: List[VRPNbH]
    ) -> List[torch.Tensor]:
        x_cat = torch.stack([x_tour, x_best_tour], dim=3)
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
        nbhs,
    ):
        """
        """
        b, _, _ = x_edges.shape
        nbh_logits = self.get_nbh_logits(
            x_edges,
            x_edges_values,
            x_nodes_coord,
            x_tour,
            x_best_tour,
            nbhs,
        )
        action_idxs, moves, pi = VRPActionNet._sample_actions(
            nbh_logits.squeeze(-1), nbhs, greedy=self.greedy
        )

        return moves, pi, action_idxs

    def get_action_pref(
        self,
        x_edges,
        x_edges_values,
        x_nodes_coord,
        x_tour,
        x_best_tour,
        nbhs,
        actions: torch.Tensor
    ):
        """
        """
        b, _, _ = x_edges.shape
        nbh_logits = self.get_nbh_logits(
            x_edges,
            x_edges_values,
            x_nodes_coord,
            x_tour,
            x_best_tour,
            nbhs,
        )
        action_idxs, moves, pi = VRPActionNet._sample_actions(
            nbh_logits.squeeze(-1), nbhs, greedy=self.greedy, actions=actions
        )

        return moves, pi, action_idxs

from deepls.VRPState import VRPState
from typing import Tuple

def model_input_from_states(states: List[VRPState], best_states: List[VRPState]):
    x_edges = []
    x_edges_values = []
    x_nodes_coord = []
    x_tour = []
    x_best_tour = []
    nbhs = []
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
        x_nodes_coord.append(torch.Tensor(state.nodes_coord).unsqueeze(0))
        nbhs.append(state.get_nbh())
    return (
        torch.cat(x_edges, dim=0),
        torch.cat(x_edges_values, dim=0),
        torch.cat(x_nodes_coord, dim=0),
        torch.cat(x_tour, dim=0),
        torch.cat(x_best_tour, dim=0),
        nbhs
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
        x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour, nbhs = \
            list(model_input_from_states(states, best_states))

        model_input = [x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour]
        with torch.no_grad():
            moves, pis, action_idxs = self.net(*[t.clone().to(self.device) for t in model_input] + [nbhs])
        cache = {
            'model_input': model_input + [nbhs], 'action': action_idxs.detach().to('cpu'),
            'action_pref': pis.detach().to('cpu'), 'tour_len': [state.get_cost() for state in states]
        }
        return moves, cache

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
        nbhs = [c[5] for c in cached_inputs]
        nbhs = [nbhs[p][0] for p in perm]
        actions = torch.cat([e['cache']['action'] for e in experiences], dim=0)[perm]
        h_sa_old = torch.cat([e['cache']['action_pref'] for e in experiences], dim=0)[perm]

        model_inputs = [x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour]
        _, h_sa, _ = self.net.get_action_pref(*[t.clone().to(self.device) for t in model_inputs] + [nbhs] + [actions.clone()])
        return h_sa, h_sa_old.to(self.device)


if __name__=="__main__":
    import numpy as np
    from deepls.agent import AverageStateRewardBaselineAgent

    episodes = 5000
    N = 10
    num_steps = 1
    hidden_dim = 16


    config = {
        "node_dim": 2,
        "voc_edges_in": 3,
        "hidden_dim": hidden_dim,
        "num_layers": 2,
        "mlp_layers": 3,
        "aggregation": "mean",
        "num_edge_cat_features": 2
    }
    # net = VRPActionNet(config)
    #
    # runner = ActionNetRunner(net, 'cpu')
    agent_config = {
        'replay_buffer_size': 100,
        'batch_sz': 64,
        'minibatch_sz': 32,
        'policy_optimize_every': 10,
        'model': {
            "node_dim": 2,
            "voc_edges_in": 3,
            "hidden_dim": hidden_dim,
            "num_layers": 2,
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
        'device': 'cpu'
    }

    from torch.optim import Adam

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

            self.net = VRPActionNet(model_config).to(device)
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
            self.greedy = False

        def compute_baseline(self, experiences, perm):
            # TODO: override this with plain old moving average
            return self.moving_avg * torch.ones(len(perm))

    agent = AverageStateRewardBaselineAgentVRP()
    agent.agent_init(agent_config)
    moving_avg = None
    print(agent.train)

    demands = np.ones(shape=(N))
    demands_norm = demands / np.sum(demands)
    coords = np.zeros(shape=(N + 1, 2))
    coords[0] = 0.5
    coords[1:] = np.random.random(size=(N, 2))
    init_state = VRPState(coords, node_demands=demands, max_tour_demand=N, id = 0)

    for episode in range(episodes):

        state = copy.deepcopy(init_state)
        state.id = episode
        best_state = copy.deepcopy((state))
        best_reward = best_state.get_cost(exclude_depot=False)
        actions = agent.agent_start([(state, best_state)])
        init_cost = state.get_cost(exclude_depot=False)
        print(' ----- state', state.get_tour_lens())

        for step in range(num_steps):
            # print(actions[0], state.all_tours_as_list())
            state.apply_move(actions[0])
            reward = state.get_cost(exclude_depot = False)
            if reward < best_reward:
                best_state = copy.deepcopy(state)
                best_reward = best_state.get_cost(exclude_depot=False)
            if step < num_steps-1:
                actions = agent.agent_step(
                    np.array([0]),
                    [(state, best_state)]
                )
            else:
                if moving_avg is None:
                    moving_avg = best_reward
                else:
                    moving_avg = 0.9 * moving_avg + 0.1 * best_reward
                print(actions, f"{reward:.3f}", f"{best_reward:.3f}")
                agent.moving_avg = moving_avg
                agent.agent_end(np.array([best_reward]))
        print(' ----- best state', best_state.get_tour_lens())
        print(f"{best_state.get_cost(exclude_depot=False):.3f} / {init_cost:.3f} = {best_state.get_cost(exclude_depot=False) / init_cost:.3f}", f"{moving_avg:.3f}")