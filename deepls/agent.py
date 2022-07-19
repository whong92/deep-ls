from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import torch
from collections import OrderedDict
from typing import Optional, Any

from deepls.gcn_model import model_input_from_states, TSPRGCNValueNet, TSPRGCNActionNet, TSPRGCNLogNormalValueNet
from deepls.graph_utils import tour_nodes_to_W
from deepls.TSP2OptEnv import TSP2OptState
from torch.optim import Adam


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
    """
    LRP implementation with an ordered dict
    """
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: Any) -> Optional[Any]:
        if key not in self.cache:
            return None
        else:
            return self.cache[key]

    def has(self, key):
        return key in self.cache

    def put(self, key: Any, value: Any) -> Any:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            return self.cache.popitem(last=False)
        return None


class ExperienceBuffer:
    def __init__(self, size):
        """
        Experience buffer for episodic MDP's
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
        This method can also handle sequences of all the following arguments
        Args:
            state (TSP2OptState): The state.
            action : Whatever is returned from agent.policy(state)
            reward (float): The reward.
            terminal (bool): 1 if the next state is a terminal state and 0 otherwise.
            next_state (TSP2OptState): The next state.
            timestep (int): the step count in the episode
            cache (Dict[str, Any]): anything that needs to be cached for this episode (so re-computation can be saved)
        """
        if not self.buffer.has(episode):
            popped_item = self.buffer.put(episode, [])
            if popped_item is not None:
                self.size -= len(popped_item[1])

        self.buffer.get(episode).append({
            'state': copy.deepcopy(state),
            'action': copy.deepcopy(action),
            'reward': copy.deepcopy(reward),
            'terminal': copy.deepcopy(terminal),
            'next_state': copy.deepcopy(next_state),
            'ts': copy.deepcopy(timestep),
            'cache': copy.deepcopy(cache)
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

    # Work Required: No.
    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Sequence[TSP2OptState]): the (multi)state from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
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
            reward (Sequence[float]): the rewards received for taking the last action taken
            state (Sequence[TSP2OptState]): the (multi)state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        self.episode_steps += 1

        if self.train:
            # Append new experience to replay buffer - it deep copies everything so no need to worry about it
            self.replay_buffer.append(
                self.episode,
                self.last_state,
                self.last_action,
                reward,
                False,
                state,
                self.episode_steps,
                self.last_cache
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
            reward (Sequence[float]): the rewards the agent received for entering the
                terminal state.
        """
        self.episode_steps += 1

        if self.train:
            terminal_state = None
            # Append new experience to replay buffer
            self.replay_buffer.append(
                self.episode,
                self.last_state,
                self.last_action,
                reward,
                True,
                terminal_state,
                self.episode_steps,
                self.last_cache
            )

            # compute returns - no discounting for now
            experience = self.replay_buffer.get_episode(self.episode)
            returns = torch.flip(
                torch.cumsum(
                    torch.as_tensor([step['reward'] for step in experience][::-1]),
                    dim=0
                ),
                dims=[0]
            )  # num_steps x batch_size
            for ret, step in zip(returns, experience):
                step['cache']['return'] = ret
                step['cache']['average_return'] = torch.mean(ret) * torch.ones_like(ret)

            # Perform replay steps:
            experiences_dict = self.replay_buffer.sample_experience(self.replay_buffer.get_size())
            self.agent_optimize(experiences_dict)


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

    def policy(self, states):
        """
        run this with provided state to get action
        """
        states = [state[0] for state in states]
        actions = []
        for state in states:
            e0, e1 = _make_random_move(state)
            action = {'terminate': False, 'e0': e0, 'e1': e1}
            actions.append(action)
        return actions, {}

    def agent_optimize(self, experiences):
        """
        run this with provided experiences to run one step of optimization
        """
        pass


class ActionNetRunner:
    """
    wraps a bunch of methods that can be re-used to run the policy
    """
    def __init__(self, net: TSPRGCNActionNet, device):
        self.net = net
        self.device = device

    def policy(self, states):
        """
        :param states: sequence of tuple of 2 TSP2OptEnv states - the current state and the best state so far
        :return:
            actions - the actions sampled for this set of states
            cache - stuff that the policy net expects to be cached (to avoid re-computation), and returned to it in
            the list of experiences which is given in e.g. get_action_pref method
        """
        best_states = [state[1] for state in states]
        states = [state[0] for state in states]
        # cur state
        x_edges, x_edges_values, x_nodes_coord, x_tour = list(model_input_from_states(states))
        # best_state
        _, _, _, x_best_tour = model_input_from_states(best_states)
        # get x_tour_directed
        x_tour_directed = torch.stack([
            torch.as_tensor(tour_nodes_to_W(state.tour_nodes, directed=True))
            for state in states
        ], dim=0)

        model_input = [x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour, x_tour_directed]
        with torch.no_grad():
            edges, pis, action_idxs = self.net(*[t.clone().to(self.device) for t in model_input])
        edges = edges.detach().to('cpu')
        # edge_0, edge_1 = edges[0, :, 1:]
        edges_0 = edges[:, 0, 1:]
        edges_1 = edges[:, 1, 1:]
        actions = [
            {'terminate': False, 'e0': edge_0, 'e1': edge_1}
            for edge_0, edge_1 in zip(edges_0, edges_1)
        ]
        cache = {
            'model_input': model_input, 'action': action_idxs.detach().to('cpu'),
            'action_pref': pis.detach().to('cpu'), 'tour_len': [state.tour_len for state in states]
        }
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
        x_tour_directed = torch.cat([c[5] for c in cached_inputs], dim=0)[perm]
        actions = torch.cat([e['cache']['action'] for e in experiences], dim=0)[perm]

        model_inputs = [x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour, actions, x_tour_directed]
        _, h_sa, _ = self.net.get_action_pref(*[t.clone().to(self.device) for t in model_inputs])
        return h_sa


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
        self.action_net_runner = ActionNetRunner(self.net, device)

        self.value_net_type = model_config.get('value_net_type', 'normal')
        if self.value_net_type == 'normal':
            self.critic_baseline = TSPRGCNValueNet(model_config).to(self.device)
            self.critic_loss = torch.nn.HuberLoss(delta=0.2).to(self.device)
        elif self.value_net_type == 'lognormal':
            self.critic_baseline = TSPRGCNLogNormalValueNet(model_config).to(self.device)
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
        self.greedy = False

    def compute_baseline_and_loss(self, experiences, perm, returns):
        cached_inputs = [e['cache']['model_input'] for e in experiences]
        x_edges = torch.cat([c[0] for c in cached_inputs], dim=0)[perm]
        x_edges_values = torch.cat([c[1] for c in cached_inputs], dim=0)[perm]
        x_nodes_coord = torch.cat([c[2] for c in cached_inputs], dim=0)[perm]
        x_tour = torch.cat([c[3] for c in cached_inputs], dim=0)[perm]
        x_best_tour = torch.cat([c[4] for c in cached_inputs], dim=0)[perm]
        model_inputs = [x_edges, x_edges_values, x_nodes_coord, x_tour, x_best_tour]

        baseline, critic_loss = None, None
        if self.value_net_type == 'normal':
            baseline = self.critic_baseline(*[t.clone().to(self.device) for t in model_inputs])
            critic_loss = self.critic_loss(baseline, returns)
        elif self.value_net_type == 'lognormal':
            baseline_dist: torch.distributions.LogNormal = \
                self.critic_baseline(*[t.clone().to(self.device) for t in model_inputs])
            baseline = -baseline_dist.mean
            critic_loss = -baseline_dist.log_prob(-returns).mean()

        return baseline, critic_loss

    def set_greedy(self, greedy=False):
        self.greedy = greedy
        self.net.set_greedy(greedy)

    def policy(self, states):
        """
        run this with provided state to get action
        """
        return self.action_net_runner.policy(states)

    def agent_optimize(self, experiences):
        """
        run this with provided experiences to run one step of optimization
        """
        optimize_policy = (self.episode % self.policy_optimize_every == 0) \
                          and (self.episode > self.dont_optimize_policy_steps)
        optimize_critic = self.episode % self.critic_optimize_every == 0

        returns = torch.cat([e['cache']['return'] for e in experiences], dim=0).float()
        if len(returns) < self.batch_size:
            return
        perm = np.random.choice(len(returns), self.batch_size, replace=False)
        returns = returns[perm]

        self.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        for (minibatch_perm, minibatch_returns), weight in \
                iter_over_tensor_minibatch(perm, returns, minibatch_size=self.minibatch_size):

            minibatch_baseline = None
            critic_loss = None
            if optimize_critic or optimize_policy:
                _returns = minibatch_returns.clone().to(self.device)
                if optimize_critic:
                    minibatch_baseline, critic_loss = self.compute_baseline_and_loss(experiences, minibatch_perm, _returns)
                    critic_loss.backward()
                else:
                    with torch.no_grad():
                        minibatch_baseline, critic_loss = self.compute_baseline_and_loss(experiences, minibatch_perm, _returns)

            # TODO: use this to deal with accumulating gradients
            if optimize_policy:
                h_sa = self.action_net_runner.get_action_pref(experiences, minibatch_perm)
                eligibility_loss = -h_sa  # NLL loss
                td_err = minibatch_returns.clone().to(self.device) - minibatch_baseline.detach()
                policy_loss = (td_err.to(self.device) * eligibility_loss).mean()  # TODO: discounting!(?)
                policy_loss.backward()

            self.critic_loss_val = critic_loss.detach().to('cpu')
            self.critic_abs_err = torch.abs(
                minibatch_baseline.detach().to('cpu') - minibatch_returns
            ).mean()

        self.critic_optimizer.step()
        self.optimizer.step()


class AverageStateRewardBaselineAgent(REINFORCEAgent):

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
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(bla, path)

    def load(self, path, init_config=True):
        bla = torch.load(path, map_location=self.device)
        agent_config = bla['agent_config']
        agent_config['device'] = self.device
        if init_config:
            self.agent_init(agent_config)
        self.net.load_state_dict(bla['net'])
        self.optimizer.load_state_dict(bla['optimizer'])

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
        self.action_net_runner = ActionNetRunner(self.net, device)
        self.dont_optimize_policy_steps = model_config.get('dont_optimize_policy_steps', 0)
        self.critic_abs_err = None

        self.policy_optimize_every = agent_config['policy_optimize_every']
        self.minibatch_size = agent_config['minibatch_sz']
        self.greedy = False

    def compute_baseline(self, experiences, perm):
        average_returns = torch.cat([e['cache']['average_return'] for e in experiences], dim=0)[perm]
        return average_returns

    def set_greedy(self, greedy=False):
        self.greedy = greedy
        self.net.set_greedy(greedy)

    def policy(self, states):
        """
        run this with provided state to get action
        """
        return self.action_net_runner.policy(states)

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
            self.optimizer.zero_grad()
            for (minibatch_perm, minibatch_returns, minibatch_baseline), weight in \
                    iter_over_tensor_minibatch(perm, returns, baseline, minibatch_size=self.minibatch_size):
                h_sa = self.action_net_runner.get_action_pref(experiences, minibatch_perm)
                eligibility_loss = -h_sa  # NLL loss
                td_err = (minibatch_returns.clone() - minibatch_baseline.detach()).to(self.device)
                policy_loss = (td_err.to(self.device) * eligibility_loss).mean() * weight  # TODO: discounting!(?)
                policy_loss.backward()
            self.optimizer.step()


def iter_over_tensor_minibatch(*tensors, minibatch_size):
    """
    for tensors of size (batchsize, ...), break it up into minibatches of size minibatch_size and iterate over them
    also returns the weight of each minibatch defined as:
    weight_i = minibatch_i_size / batchsize
    :param tensors:
    :param minibatch_size:
    :return:
    """
    batch_size = tensors[0].shape[0]
    num_whole_batch = batch_size // minibatch_size
    has_partial_batch = (batch_size % minibatch_size) != 0
    mini_batch_weights = [minibatch_size / batch_size for _ in range(num_whole_batch)]
    mini_batch_sizes = [minibatch_size for _ in range(num_whole_batch)]
    if has_partial_batch:
        mini_batch_weights += [(batch_size % minibatch_size) / batch_size]
        mini_batch_sizes += [batch_size % minibatch_size]
    mini_batch_offsets = np.cumsum([0] + mini_batch_sizes)
    for start, end, weight in zip(mini_batch_offsets[:-1], mini_batch_offsets[1:], mini_batch_weights):
        minibatch_tensors = [tensor[start:end] for tensor in tensors]
        yield minibatch_tensors, weight
