TSP_SIZE_TO_TRAIN_DATA_F = {
    10: 'tsp10_train_concorde.txt',
    20: 'tsp20_train_concorde.txt',
    30: 'tsp30_train_concorde.txt',
    50: 'tsp50_concorde.txt',
    100: 'tsp100_concorde.txt',
}

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from tqdm import tqdm

from deepls.TSP2OptEnv import TSP2OptMultiEnv
from deepls.agent import AverageStateRewardBaselineAgent


font = cv2.FONT_HERSHEY_COMPLEX_SMALL


def moving_average(arr, c=100):
    return np.convolve(arr, np.ones(shape=(c,))/c, mode='valid')


@dataclass
class Run:
    episode: int
    run_len: int

    @property
    def run(self):
        return self.episode // self.run_len

    @property
    def frac(self):
        return self.episode % self.run_len

    @property
    def is_end(self):
        return ((self.episode + 1) % self.run_len) == 0

    @property
    def is_start(self):
        return (self.episode % self.run_len) == 0


from typing import List
@dataclass
class RunSched:
    runs: List[int]
    episode_lens: List[int]
    run_lens: List[int]

    def __post_init__(self):
        assert len(self.runs) == len(self.episode_lens) == len(self.run_lens)
        assert self.runs == sorted(self.runs)
        self.cum_episode_sched = np.cumsum(
            np.array(self.runs[1:] + [float("inf")]) * np.array(self.run_lens)
        )
        print(self.cum_episode_sched)

    def get_schedule(self, cur_run):
        for run, next_run, episode_len, run_len in \
                zip(self.runs, self.runs[1:] + [float("inf")], self.episode_lens, self.run_lens):
            if cur_run >= run and cur_run < next_run:
                return episode_len, run_len
        raise Exception("should not hit this!")


def run_experiment(
    experiment_config
):
    experiment_name = experiment_config.get('experiment_name', 'default')
    data_root = experiment_config['data_root']
    model_root = f"{experiment_config['model_root']}/{experiment_name}/"
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    problem_sz = experiment_config['problem_sz']

    run_sched = RunSched(**experiment_config['run_sched'])
    model_ckpt = experiment_config.get('model_ckpt')
    num_samples_per_instance = experiment_config['num_samples_per_instance']
    val_every = experiment_config['val_every']
    start_episode = experiment_config['start_episode']
    train_episodes = experiment_config['train_episodes']

    agent_config = experiment_config['agent_config']

    env = TSP2OptMultiEnv(
        max_num_steps=problem_sz,
        num_nodes=problem_sz,
        data_f=f'{data_root}/{TSP_SIZE_TO_TRAIN_DATA_F[problem_sz]}',
        num_samples_per_instance=num_samples_per_instance,
        num_instance_per_batch=2,
        shuffle_data=True,
        ret_log_tour_len=False
    )
    env.reset()

    agent = AverageStateRewardBaselineAgent()
    agent.agent_init(agent_config)
    if model_ckpt is not None:
        agent.load(model_ckpt, init_config=False)
        optim_config = agent_config['optim']
        for g in agent.optimizer.param_groups:
            # make sure we use the LR specified in agent config
            g['lr'] = optim_config['step_size']

    ve_error = []
    avg_train_opt_gaps = []

    agent.set_train()
    pbar = tqdm(range(start_episode, start_episode + train_episodes))
    for irun in pbar:
        episode_len, run_len = run_sched.get_schedule(irun)
        # run for run_len episodes
        for episode in range(run_len):
            run = Run(episode, run_len)  # fractional run
            if run.is_start:
                # run for episode_len steps
                env.reset(fetch_next=True, max_num_steps=episode_len)
            else:
                env.reset_episode()

            states = env.get_state()
            actions = agent.agent_start(states)
            while True:
                # Take a random action
                rets = env.step(actions)
                states = [ret[0] for ret in rets]
                rewards = [ret[1] for ret in rets]
                dones = [ret[2] for ret in rets]

                if dones[0] == True:
                    agent.agent_end(rewards)
                    # if end of epoch
                    if run.is_end:
                        print(' ------- ', episode)
                        if agent.critic_abs_err is not None:
                            ve_error.append(torch.mean(agent.critic_abs_err).detach())
                        else:
                            ve_error.append(-1)
                        avg_train_opt_gaps.append(
                            np.mean([(state[1].tour_len / state[0].opt_tour_len) - 1. for state in states])
                        )
                    break
                else:
                    actions = agent.agent_step(rewards, states)

        if irun % val_every == 0:
            agent.save(
                f'{model_root}'
                f'/model-{irun:05d}-'
                f'val-{np.mean(avg_train_opt_gaps[-val_every:]):.3f}.ckpt')

        desc = f"" \
               f"train opt gap = {np.mean(avg_train_opt_gaps[-val_every:]):.3f}  " \
               f"ve_err = {np.mean(ve_error[-val_every:]):.3f}"
        pbar.set_description(desc)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        "-d",
        default=os.path.join(os.path.dirname(__file__), 'data', 'tsp-data'),
        help="location of data files"
    )
    parser.add_argument(
        "--modelroot",
        "-m",
        default=os.path.join(os.path.dirname(__file__), 'model'),
        help="location to save model files"
    )
    args = parser.parse_args()

    agent_config = {
        'replay_buffer_size': 5,
        # below settings are used mainly in agent_optimize
        # batch_sz is the number of rollout samples used in the gradient step
        # minibatch_sz is to control parallelism
        'batch_sz': 64,
        'minibatch_sz': 32,
        # after how many episodes do we optimize policy / critic?
        'policy_optimize_every': 2,
        'critic_optimize_every': 1,
        # this doesn't work well - PPO's lower bound surrogate loss isn't as effective as the exact PG loss
        # we don't have a convergence issue anyways, so this was purely for intellectual interest
        'use_ppo_update': False,
        # architecture settings
        'model': {
            "voc_edges_in": 3,
            "hidden_dim": 128,
            "num_layers": 5,
            "mlp_layers": 3,
            "aggregation": "mean",
            "node_dim": 2,
            'dont_optimize_policy_steps': 0,
            'value_net_type': 'normal'
        },
        # optimizer settings
        'optim': {
            'step_size': 1e-5,
            'beta_m': 0.9,
            'beta_v': 0.999,
            'epsilon': 1e-8
        },
        'device': 'cuda'
    }

    experiment_config = {
        'problem_sz': 20,
        'experiment_name': 'test',
        'model_ckpt': None,
        'num_samples_per_instance': 12,
        'val_every': 500,
        'start_episode': 0,
        'train_episodes': 20000,
        # schedule for 20 nodes
        'run_sched': {
            'runs': [0, 5000, 10000, 15000],
            'episode_lens': [4, 8, 20, 40],
            'run_lens': [5, 5, 2, 2],
        },
        'agent_config': agent_config,
        'model_root': args.modelroot,
        'data_root': args.dataroot
    }

    run_experiment(experiment_config)