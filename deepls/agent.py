"""An abstract class that specifies the Agent API for RL-Glue-py.
"""

from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import torch
from collections import OrderedDict
from typing import Optional, Any

from deepls.gcn_model import model_input_from_states
from deepls.TSP2OptEnv import TSP2OptState


class BaseAgent:
    """Implements the agent for an RL-Glue environment.
    Note:
        agent_init, agent_start, agent_step, agent_end, agent_cleanup, and
        agent_message are required methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""

    @abstractmethod
    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """

    @abstractmethod
    def agent_cleanup(self):
        """Cleanup done after the agent ends."""

    @abstractmethod
    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """


class LRPCache:

    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    # we return the value of the key
    # that is queried in O(1) and return -1 if we
    # don't find the key in out dict / cache.
    # And also move the key to the end
    # to show that it was recently used.
    def get(self, key: Any) -> Optional[Any]:
        if key not in self.cache:
            return None
        else:
            return self.cache[key]

    def has(self, key):
        return key in self.cache

    # first, we add / update the key by conventional methods.
    # And also move the key to the end to show that it was recently used.
    # But here we will also check whether the length of our
    # ordered dictionary has exceeded our capacity,
    # If so we remove the first key (least recently used)
    def put(self, key: Any, value: Any) -> Any:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            return self.cache.popitem(last=False)
        return None


class ExperienceBuffer:
    def __init__(self, size):
        """
        Args:
            size (integer): The size of the replay buffer in terms of number of episodes
        """
        # episode -> experience map
        self.buffer = LRPCache(size)
        self.max_size = size
        self.num_steps = [0]
        self.size = 0  # total number of steps

    def append(self, episode, state, action, reward, terminal, next_state, timestep, cache):
        """
        Args:
            state (Numpy array): The state.
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state.
        """
        if not self.buffer.has(episode):
            popped_item = self.buffer.put(episode, [])
            if popped_item is not None:
                self.size -= len(popped_item[1])

        self.buffer.get(episode).append({
            'state': state,
            'action': action,
            'reward': reward,
            'terminal': terminal,
            'next_state': next_state,
            'ts': timestep,
            'cache': cache
        })
        self.size += 1

    def flush(self):
        """
        empty the buffer
        """
        self.buffer = LRPCache(self.max_size)

    def get_size(self):
        return self.size

    def get_episode(self, episode):
        return self.buffer.get(episode)

    def sample_experience(self, minibatch_sz):
        sample_coords = []
        # I know this isn't the most efficient way of sampling things, but I can assure you that
        # this won't be the bottleneck in the training process
        for episode, experiences in self.buffer.cache.items():
            sample_coords.extend([(episode, step) for step in range(len(experiences))])
        sample_coords = np.array(sample_coords)  # N x 2
        sample_coords = sample_coords[
            np.random.choice(len(sample_coords), size=minibatch_sz, replace=False)
        ]
        sampled_experiences = []
        for episode, step in sample_coords:
            sampled_experiences.append(self.buffer.get(episode)[step])
        return sampled_experiences


class REINFORCEAgent(BaseAgent):
    __metaclass__ = ABCMeta

    def init_replay_buffer(self, replay_buffer_size):
        # initialize replay buffer
        self.replay_buffer = ExperienceBuffer(replay_buffer_size)

    @abstractmethod
    def _agent_init(self, agent_config):
        pass

    # Work Required: No.
    def agent_init(self, agent_config={}):
        """Setup variables to track state
        """
        self.last_state = None
        self.last_action = None
        self.last_cache = None  # anything that needs caching for timestep t-1

        self.sum_rewards = 0
        self.episode_steps = 0
        self.episode = -1

        self.train = True  # whether or not to perform optimization

        self._agent_init(agent_config)
        self.init_replay_buffer(agent_config['replay_buffer_size'])
        self.minibatch_size = agent_config['minibatch_sz']

    # Work Required: No.
    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.sum_rewards = 0
        self.episode_steps = 0
        self.episode += 1
        self.last_state = copy.deepcopy(state)
        self.last_action, self.last_cache = self.policy(self.last_state)
        return self.last_action

    @abstractmethod
    def policy(self, state):
        """
        run this with provided state to get action
        """
        raise NotImplementedError

    @abstractmethod
    def agent_optimize(self, experiences):
        """
        run this with provided experiences to run one step of optimization
        """
        raise NotImplementedError

    # weights update using optimize_network, and updating last_state and last_action (~5 lines).
    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        self.episode_steps += 1

        # Append new experience to replay buffer
        # TODO: deepcopy in replay buffer instead
        self.replay_buffer.append(
            self.episode,
            copy.deepcopy(self.last_state),
            copy.deepcopy(self.last_action),
            reward,
            False,
            copy.deepcopy(state),
            self.episode_steps,
            copy.deepcopy(self.last_cache)
        )
        # Select action
        action, cache = self.policy(state)

        # Update the last state and last action.
        self.last_state = copy.deepcopy(state)
        self.last_action = copy.deepcopy(action)
        self.last_cache = copy.deepcopy(cache)

        return action

    def set_train(self):
        self.train = True

    def set_eval(self):
        self.train = False

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1

        # TODO: figure out terminal state rep
        # for now don't store terminal state, but propagate its return
        terminal_state = None
        # Append new experience to replay buffer
        #         self.replay_buffer.append(
        #             self.episode,
        #             copy.deepcopy(self.last_state),
        #             copy.deepcopy(self.last_action),
        #             reward,
        #             True,
        #             terminal_state,
        #             self.episode_steps,
        #             copy.deepcopy(self.last_cache)
        #         )

        # compute rewards
        experience = self.replay_buffer.get_episode(self.episode)
        returns = torch.flip(
            torch.cumsum(torch.Tensor([reward] + [step['reward'] for step in experience][::-1]), dim=0), dims=[0])
        for ret, step in zip(returns, experience):
            step['cache']['return'] = ret.item()

        if self.train:
            # Perform replay steps:
            if self.replay_buffer.get_size() >= self.minibatch_size:
                # TODO: maybe run multiple optimize steps?
                # Get sample experiences from the replay buffer and optimize
                experiences_dict = self.replay_buffer.sample_experience(self.minibatch_size)
                self.agent_optimize(experiences_dict)

                # TODO: maybe don't flush and tolerate stale episodes? Use off-policy updates maybe
                # self.replay_buffer.flush()


def _make_random_move(state: TSP2OptState):
    i, j = np.where(state.tour_adj)
    u = i[i < j]
    v = j[i < j]
    es = np.stack([u, v])
    e1, e2 = np.random.choice(es.shape[1], 2)
    e1 = es[:, e1]
    e2 = es[:, e2]
    return e1, e2


class RandomActionREINFORCEAgent(REINFORCEAgent):

    def policy(self, state):
        """
        run this with provided state to get action
        """
        e0, e1 = _make_random_move(state)
        action = {'terminate': False, 'e0': e0, 'e1': e1}
        model_input = model_input_from_states([state])
        return action, {'model_input': model_input}

    def agent_optimize(self, experiences):
        """
        run this with provided experiences to run one step of optimization
        """
        pass


from torch.optim import Adam
from deepls.gcn_model import TSPRGCNValueNet, TSPRGCNActionNet
from deepls.graph_utils import tour_nodes_to_W

class GRCNCriticBaselineAgent(REINFORCEAgent):

    def set_train(self):
        super().set_train()
        self.net.train()
        self.critic_baseline.train()

    def set_eval(self):
        super().set_eval()
        self.net.eval()
        self.critic_baseline.eval()

    def save(self, path):
        bla = {
            'agent_config': self.agent_config,
            'net': self.net.state_dict(),
            'critic': self.critic_baseline.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        torch.save(bla, path)

    def load(self, path, init_config=True):
        bla = torch.load(path, map_location=self.device)
        agent_config = bla['agent_config']
        agent_config['device'] = self.device
        if init_config:
            self.agent_init(agent_config)
        self.net.load_state_dict(bla['net'])
        self.critic_baseline.load_state_dict(bla['critic'])
        self.optimizer.load_state_dict(bla['optimizer'])
        self.critic_optimizer.load_state_dict(bla['critic_optimizer'])

    def _agent_init(self, agent_config):
        self.agent_config = copy.deepcopy(agent_config)
        model_config = agent_config['model']
        optimizer_config = agent_config['optim']
        device = agent_config.get('device', 'cpu')
        self.device = device

        self.net = TSPRGCNActionNet(model_config).to(device)
        self.optimizer = Adam(
            self.net.parameters(),
            lr=optimizer_config['step_size'],
            betas=(optimizer_config['beta_m'], optimizer_config['beta_v']),
            eps=optimizer_config['epsilon']
        )

        self.critic_baseline = TSPRGCNValueNet(agent_config['model']).to(self.device)
        self.critic_optimizer = Adam(
            self.critic_baseline.parameters(),
            lr=optimizer_config['step_size_critic'],
            betas=(optimizer_config['beta_m'], optimizer_config['beta_v']),
            eps=optimizer_config['epsilon']
        )
        self.dont_optimize_policy_steps = model_config.get('dont_optimize_policy_steps', 0)

        self.critic_loss = torch.nn.HuberLoss(delta=0.2).to(self.device)
        # torch.nn.SmoothL1Loss().to(self.device)
        # torch.nn.MSELoss().to(self.device)
        self.critic_loss_val = None
        self.critic_abs_err = None

        self.policy_optimize_every = agent_config['policy_optimize_every']
        self.critic_optimize_every = agent_config['critic_optimize_every']
        self.greedy = False

    def compute_baseline(self, experiences):
        cached_inputs = [e['cache']['model_input'] for e in experiences]
        x_edges = torch.cat([c[0] for c in cached_inputs], dim=0)
        x_edges_values = torch.cat([c[1] for c in cached_inputs], dim=0)
        x_nodes_coord = torch.cat([c[2] for c in cached_inputs], dim=0)
        x_tour = torch.cat([c[3] for c in cached_inputs], dim=0)
        x_best_tour = torch.cat([c[4] for c in cached_inputs], dim=0)
        model_inputs = [x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour]
        # model_inputs = [x_edges, x_edges_values, x_nodes_coord, torch.ones_like(x_tour), torch.ones_like(x_best_tour)]
        return self.critic_baseline(*[t.clone().to(self.device) for t in model_inputs])
        # return torch.zeros(len(cached_inputs))

    def get_action_pref(self, experiences):
        cached_inputs = [e['cache']['model_input'] for e in experiences]
        x_edges = torch.cat([c[0] for c in cached_inputs], dim=0)
        x_edges_values = torch.cat([c[1] for c in cached_inputs], dim=0)
        x_nodes_coord = torch.cat([c[2] for c in cached_inputs], dim=0)
        x_tour = torch.cat([c[3] for c in cached_inputs], dim=0)
        x_best_tour = torch.cat([c[4] for c in cached_inputs], dim=0)
        x_tour_directed = torch.cat([c[5] for c in cached_inputs], dim=0)
        actions = torch.cat([e['cache']['action'] for e in experiences], dim=0)

        # print(torch.as_tensor([e['reward'] for e in experiences]))
        model_inputs = [x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour, actions, x_tour_directed]
        _, h_sa, _ = self.net.get_action_pref(*[t.clone().to(self.device) for t in model_inputs])
        return h_sa

    def set_greedy(self, greedy=False):
        self.greedy = greedy
        self.net.set_greedy(greedy)

    def policy(self, state):
        """
        run this with provided state to get action
        """
        state, best_state = state
        # cur state
        x_edges, x_edges_values, x_nodes_coord, x_tour = list(model_input_from_states([state]))
        # best_state
        _, _, _, x_best_tour = model_input_from_states([best_state])
        # get x_tour_directed
        x_tour_directed = torch.as_tensor(tour_nodes_to_W(state.tour_nodes, directed=True)).unsqueeze(0)

        model_input = [x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour, x_tour_directed]
        with torch.no_grad():
            edges, pis, action_idxs = self.net(*[t.clone().to(self.device) for t in model_input])
        edges = edges.detach().to('cpu')
        edge_0, edge_1 = edges[0, :, 1:]
        action = {'terminate': False, 'e0': edge_0, 'e1': edge_1}
        return action, {'model_input': model_input, 'action': action_idxs.detach().to('cpu'),
                        'action_pref': pis.detach().to('cpu'), 'tour_len': state.tour_len}

    def agent_optimize(self, experiences):
        """
        run this with provided experiences to run one step of optimization
        """
        optimize_policy = (self.episode % self.policy_optimize_every == 0) \
                          and (self.episode > self.dont_optimize_policy_steps)
        optimize_critic = self.episode % self.critic_optimize_every == 0
        if optimize_policy or optimize_critic:
            returns = torch.as_tensor([e['cache']['return'] for e in experiences])
            baseline = self.compute_baseline(experiences)

        if optimize_policy:
            h_sa = self.get_action_pref(experiences)
            eligibility_loss = -h_sa  # NLL loss
            td_err = returns.clone().to(self.device) - baseline.detach()
            # print(baseline[0], returns[0])
            policy_loss = (td_err.to(self.device) * eligibility_loss).mean()  # TODO: discounting!(?)
            policy_loss.backward()
            self.optimizer.step()

        if optimize_critic:
            _returns = returns.clone().to(self.device)
            critic_loss = self.critic_loss(baseline, _returns)
            self.critic_loss_val = critic_loss.detach().to('cpu')
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_abs_err = torch.abs(baseline - _returns).detach().to('cpu')


class GRCNRollingBaselineAgent(REINFORCEAgent):

    def set_train(self):
        super().set_train()
        self.net.train()

    def set_eval(self):
        super().set_eval()
        self.net.eval()

    def save(self, path):
        bla = {
            'agent_config': self.agent_config,
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'critic_baseline': self.critic_baseline,
            'step_size_critic': self.step_size_critic
        }
        torch.save(bla, path)

    def load(self, path, init_config=True):
        bla = torch.load(path, map_location=self.device)
        agent_config = bla['agent_config']
        agent_config['device'] = self.device
        if init_config:
            self.agent_init(agent_config)
        self.net.load_state_dict(bla['net'])
        self.critic_baseline = bla['critic_baseline']
        self.step_size_critic = bla['step_size_critic']
        self.optimizer.load_state_dict(bla['optimizer'])

    def _agent_init(self, agent_config):
        self.agent_config = copy.deepcopy(agent_config)
        model_config = agent_config['model']
        optimizer_config = agent_config['optim']
        device = agent_config.get('device', 'cpu')
        self.device = device

        self.dont_optimize_policy_steps = model_config.get('dont_optimize_policy_steps', 0)

        self.steps = 0
        self.net = TSPRGCNActionNet(model_config).to(device)
        self.optimizer = Adam(
            self.net.parameters(),
            lr=optimizer_config['step_size'],
            betas=(optimizer_config['beta_m'], optimizer_config['beta_v']),
            eps=optimizer_config['epsilon']
        )
        self.critic_baseline = 0.
        self.critic_loss_val = 0.
        self.critic_abs_err = 0.
        self.step_size_critic = optimizer_config['step_size_critic']

    def compute_baseline(self, experiences):
        return (torch.ones(len(experiences)) * self.critic_baseline).to(self.device)

    def get_action_pref(self, experiences):
        cached_inputs = [e['cache']['model_input'] for e in experiences]
        x_edges = torch.cat([c[0] for c in cached_inputs], dim=0)
        x_edges_values = torch.cat([c[1] for c in cached_inputs], dim=0)
        x_nodes_coord = torch.cat([c[2] for c in cached_inputs], dim=0)
        x_tour = torch.cat([c[3] for c in cached_inputs], dim=0)
        x_best_tour = torch.cat([c[4] for c in cached_inputs], dim=0)
        x_tour_directed = torch.cat([c[5] for c in cached_inputs], dim=0)
        actions = torch.cat([e['cache']['action'] for e in experiences], dim=0)

        # print(torch.as_tensor([e['reward'] for e in experiences]))
        model_inputs = [x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour, actions, x_tour_directed]
        _, h_sa, _ = self.net.get_action_pref(*[t.clone().to(self.device) for t in model_inputs])
        return h_sa

    def policy(self, state):
        """
        run this with provided state to get action
        """
        state, best_state = state
        # cur state
        x_edges, x_edges_values, x_nodes_coord, x_tour = list(model_input_from_states([state]))
        # best_state
        _, _, _, x_best_tour = model_input_from_states([best_state])
        # get x_tour_directed
        x_tour_directed = torch.as_tensor(tour_nodes_to_W(state.tour_nodes, directed=True)).unsqueeze(0)

        model_input = [x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour, x_tour_directed]
        with torch.no_grad():
            edges, pis, action_idxs = self.net(*[t.clone().to(self.device) for t in model_input])
        edges = edges.detach().to('cpu')
        edge_0, edge_1 = edges[0, :, 1:]
        action = {'terminate': False, 'e0': edge_0, 'e1': edge_1}
        return action, {'model_input': model_input, 'action': action_idxs.detach().to('cpu'),
                        'action_pref': pis.detach().to('cpu'), 'tour_len': state.tour_len}

    def agent_optimize(self, experiences):
        """
        run this with provided experiences to run one step of optimization
        """
        returns = torch.as_tensor([e['cache']['return'] for e in experiences])
        if self.episode <= self.dont_optimize_policy_steps:
            if self.episode == 0:
                self.critic_baseline = returns.mean()
                self.critic_loss_val = 0
            else:
                step_size = 1. / (self.episode)
                self.critic_baseline = returns.mean() * step_size + (1. - step_size) * self.critic_baseline
                self.critic_loss_val = self.critic_baseline - returns.mean()

        baseline = self.compute_baseline(experiences)

        # optimize policy
        h_sa = self.get_action_pref(experiences)
        eligibility_loss = -h_sa  # NLL loss
        td_err = returns.clone().to(self.device) - baseline.detach()
        # print(baseline[0], returns[0])
        policy_loss = (td_err.to(self.device) * eligibility_loss).mean()  # TODO: discounting!(?)
        policy_loss.backward()
        self.optimizer.step()

        rb_step_size = self.step_size_critic
        # optimize critic
        self.critic_baseline = self.critic_baseline * (1 - rb_step_size) + rb_step_size * returns.mean()
        self.critic_loss_val = torch.abs(self.critic_baseline - returns.mean())
        self.critic_abs_err = self.critic_loss_val