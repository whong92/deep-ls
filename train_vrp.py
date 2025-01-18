import argparse
import os
from dataclasses import dataclass
import wandb

import cv2
import numpy as np
import torch
from tqdm import tqdm

from deepls.VRPState import VRPMultiRandomEnv, plot_state, VRPMultiFileEnv, VRPReward, VRPInitTour, VRPMultiFileEnvSingleProc, VectorizedState
from deepls.vrp_gcn_model import (
    AverageStateRewardBaselineAgentVRP, VRP_STANDARD_PROBLEM_CONF, CriticBaselineAgentVRP
)
import multiprocessing
from typing import List, Union, Optional, Any

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


VRP_SIZE_TO_TRAIN_DATA_F = {
    10: 'size-10/vrp_data_with_results_train.pkl',
    20: 'size-20/vrp_data_with_results_train.pkl',
    50: 'size-50/vrp_data_with_results.pkl',
    100: 'size-100/vrp_data_with_results.pkl'
}

VRP_SIZE_TO_VAL_DATA_F = {
    10: 'size-10/vrp_data_with_results_val.pkl',
    20: 'size-20/vrp_data_with_results_val.pkl',
    50: 'size-50/vrp_data_with_results.pkl',
    100: 'size-100/vrp_data_with_results.pkl'
}

VRP_SIZE_TO_RUN_SCHED = {
    10: {
        'runs': [0, 2000, 5000],
        'episode_lens': [5, 5, 10],
        'run_lens': [2, 4, 2],
    },
    20: {
        'runs': [0, 4000],
        'episode_lens': [5, 10],
        'run_lens': [4, 4],
    },
    # schedule for 50 nodes (after 20 node pretrain)
    50: {
        'runs': [0, 5000],
        'episode_lens': [10, 20],
        'run_lens': [5, 5],
    },
    100: {
        'runs': [0, 2500, 5000],
        'episode_lens': [10, 10, 20],
        'run_lens': [10, 15, 10],
    },
}

VRP_SIZE_TO_RUN_SCHED_SS = {
    10: {
        'runs': [0],
        'episode_lens': [10],
        'run_lens': [2],
    },
    20: {
        'runs': [0],
        'episode_lens': [20],
        'run_lens': [2],
    },
    # schedule for 50 nodes (after 20 node pretrain)
    50: {
        'runs': [0],
        'episode_lens': [10],
        'run_lens': [20],
    },
    100: {
        'runs': [0],
        'episode_lens': [10],
        'run_lens': [40],
    },
}


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

    def get_schedule(self, cur_run):
        for run, next_run, episode_len, run_len in \
                zip(self.runs, self.runs[1:] + [float("inf")], self.episode_lens, self.run_lens):
            if cur_run >= run and cur_run < next_run:
                return episode_len, run_len
        raise Exception("should not hit this!")


def run_train_run(
    agent: AverageStateRewardBaselineAgentVRP,
    irun: int,
    run_sched: RunSched,
    env: Union[VRPMultiFileEnvSingleProc, VRPMultiFileEnv],
    wandb_module: Any = None
):
    agent.set_train()
    episode_len, run_len = run_sched.get_schedule(irun)

    for episode in range(run_len):
        run = Run(episode, run_len)
        if run.is_start:
            # run for episode_len steps
            env.reset(fetch_next=True, max_num_steps=episode_len)
        else:
            env.reset_episode()

        states = env.get_state()
        actions = agent.agent_start(states, env)
        while True:
            # Take a random action
            # this return format is different from TSP's
            states, rewards, dones = env.step(actions)
            if dones[0] == True:
                metrics = agent.agent_end(rewards)
                if metrics and wandb_module is not None:
                    wandb_module.log(metrics)
                # if end of epoch
                if run.is_end:
                    if agent.critic_abs_err is not None:
                        ve_error = float(torch.mean(agent.critic_abs_err).detach())
                    else:
                        ve_error = -1
                    opt_cost = states[0].states_opt_cost[0]
                    train_opt_gaps = np.array([state.best_states_cost[0] / opt_cost - 1. for state in states])
                    train_opt_gap = np.mean(train_opt_gaps)
                    train_opt_gap_std = np.std(train_opt_gaps)
                    if wandb_module is not None:
                        wandb_module.log({'opt_gap_mean': train_opt_gap, 'opt_gap_std': train_opt_gap_std})
                    return train_opt_gap, ve_error
                break
            else:
                actions = agent.agent_step(rewards, states, env)

    assert False


def run_eval_loop(
    agent: AverageStateRewardBaselineAgentVRP,
    env: Union[VRPMultiFileEnvSingleProc, VRPMultiFileEnv],
    episodes: int
):
    # re-initialize to get the same start state
    env.init()
    agent.set_eval()
    pbar = tqdm(range(episodes))

    best_opts = []

    for _ in pbar:
        step = 0
        env.reset(fetch_next=True)
        states: List[VectorizedState] = env.get_state()
        actions = agent.agent_start(states, env)
        opt_cost = states[0].states_opt_cost[0]

        while True:
            step += 1
            states, rewards, dones = env.step(actions)
            done = dones[0]
            if done:
                best_opts.append(np.min([state.best_states_cost[0] / opt_cost - 1. for state in states]))
                break
            else:
                actions = agent.agent_step(
                    rewards,
                    states,
                    env
                )


    return np.mean(best_opts)


def run_experiment(
    experiment_config,
    wandb_module=None,
    multiproc=False
):
    experiment_name = experiment_config.get('experiment_name', 'default')
    problem_sz = experiment_config['problem_sz']

    val_data_f = f"{experiment_config['data_root']}/{VRP_SIZE_TO_VAL_DATA_F[problem_sz]}"
    train_data_f = f"{experiment_config['data_root']}/{VRP_SIZE_TO_TRAIN_DATA_F[problem_sz]}"
    model_root = f"{experiment_config['model_root']}/{experiment_name}/"
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    max_tour_demand = VRP_STANDARD_PROBLEM_CONF[problem_sz]['capacity']

    ramp_up = experiment_config.get('ramp_up', True)
    if ramp_up:
        run_sched = RunSched(**VRP_SIZE_TO_RUN_SCHED[problem_sz])
    else:
        run_sched = RunSched(**VRP_SIZE_TO_RUN_SCHED_SS[problem_sz])

    model_ckpt = experiment_config.get('model_ckpt')
    num_samples_per_instance = experiment_config['num_samples_per_instance']
    num_instance_per_batch = experiment_config['num_instance_per_batch']
    val_every = experiment_config['val_every']
    start_run = experiment_config['start_run']
    train_runs = experiment_config['train_runs']
    reward_mode = experiment_config['reward_mode']
    initializer = experiment_config['initializer']
    best_cost_eta = experiment_config.get('best_cost_eta', 1.0)

    agent_config = experiment_config['agent_config']

    if not multiproc:
        env = VRPMultiFileEnvSingleProc(
            data_f=train_data_f,
            num_nodes=problem_sz,
            max_num_steps=problem_sz,
            max_tour_demand=max_tour_demand,
            num_samples_per_instance=num_samples_per_instance,
            num_instance_per_batch=num_instance_per_batch,
            reward_mode=reward_mode,
            initializer=initializer,
            vectorize_state=True,
            best_cost_eta=best_cost_eta
        )
        env_val = VRPMultiFileEnvSingleProc(
            data_f=val_data_f,
            num_nodes=problem_sz,
            max_num_steps=problem_sz * 2,
            max_tour_demand=max_tour_demand,
            num_samples_per_instance=num_samples_per_instance,
            num_instance_per_batch=1,
            reward_mode=reward_mode,
            initializer=initializer,
            vectorize_state=True,
            best_cost_eta=best_cost_eta
        )
    else:
        num_proc = multiprocessing.cpu_count()
        env = VRPMultiFileEnv(
            data_f=train_data_f,
            num_nodes=problem_sz,
            max_num_steps=problem_sz,
            max_tour_demand=max_tour_demand,
            num_samples_per_instance=num_samples_per_instance,
            num_instance_per_batch=num_instance_per_batch,
            reward_mode=reward_mode,
            initializer=initializer,
            vectorize_state=True,
            num_proc=num_proc,
            best_cost_eta=best_cost_eta
        )
        env_val = VRPMultiFileEnv(
            data_f=val_data_f,
            num_nodes=problem_sz,
            max_num_steps=problem_sz * 2,
            max_tour_demand=max_tour_demand,
            num_samples_per_instance=num_samples_per_instance,
            num_instance_per_batch=1,
            reward_mode=reward_mode,
            initializer=initializer,
            vectorize_state=True,
            num_proc=num_proc,
            best_cost_eta=best_cost_eta
        )

    agent = AverageStateRewardBaselineAgentVRP()
    # agent = CriticBaselineAgentVRP()
    agent.agent_init(agent_config)
    if model_ckpt is not None:
        agent.load(model_ckpt, init_config=False)
        optim_config = agent_config['optim']
        for g in agent.optimizer.param_groups:
            # make sure we use the LR specified in agent config
            g['lr'] = optim_config['step_size']

    ve_error = []
    avg_train_opt_gaps = []
    avg_train_opt_gaps_ma = []

    pbar = tqdm(range(start_run, start_run + train_runs))

    moving_avg_train_opt_gap = None

    for irun in pbar:
        _train_opt_gap, _ve_error = run_train_run(agent, irun, run_sched, env)
        ve_error.append(_ve_error)
        avg_train_opt_gaps.append(_train_opt_gap)
        if moving_avg_train_opt_gap is None:
            moving_avg_train_opt_gap = _train_opt_gap
        else:
            moving_avg_train_opt_gap = 0.9 * moving_avg_train_opt_gap + 0.1 * _train_opt_gap

        if (irun % val_every == 0) and (irun > 0):
            val_opt_gap = run_eval_loop(agent, env_val, episodes=30)
            agent.save(
                f'{model_root}'
                f'/model-{irun:05d}-'
                f'val-{val_opt_gap:.3f}.ckpt')

        desc = f"" \
               f"train opt gap = {np.mean(avg_train_opt_gaps[-val_every:]):.3f}  " \
               f"ve_err = {np.mean(ve_error[-val_every:]):.3f}"
        pbar.set_description(desc)

    import matplotlib.pyplot as plt
    plt.plot(avg_train_opt_gaps_ma)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        "-d",
        default=os.path.join(os.path.dirname(__file__), 'data', 'vrp-data'),
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
        'replay_buffer_size': 10,
        # below settings are used mainly in agent_optimize
        # batch_sz is the number of rollout samples used in the gradient step
        # minibatch_sz is to control parallelism
        'batch_sz': 64,
        'minibatch_sz': 32,
        # after how many episodes do we optimize policy / critic?
        'policy_optimize_every': 1,
        'critic_optimize_every': 1,
        # this doesn't work well - PPO's lower bound surrogate loss isn't as effective as the exact PG loss
        # we don't have a convergence issue anyways, so this was purely for intellectual interest
        'use_ppo_update': False,
        # use for initial pre-train only
        'entropy_bonus': 0.002,
        'gamma': 0.99,
        # architecture settings
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
            'step_size': 1e-5,
            'step_size_critic': 2e-4,
            'beta_m': 0.9,
            'beta_v': 0.999,
            'epsilon': 1e-8
        },
        'device': 'cuda'
    }

    experiment_config = {
        'ramp_up': False,
        'problem_sz': 10,
        'experiment_name': '10-nodes-profiling',
        'model_ckpt': None, # f'{args.modelroot}/vrp-50-nodes-lr-2e-6-beta-1e-2-final-cost-singleton-init-from-scratch/model-02000-val-0.168.ckpt', # f'{args.modelroot}/vrp-50-nodes-chunked-episodes-cost-emb-delta-cost-longer-eps/model-03000-val-0.094.ckpt',
        'num_samples_per_instance': 12,
        'num_instance_per_batch': 1,
        'reward_mode': VRPReward.FINAL_COST,
        'initializer': VRPInitTour.SINGLETON,
        'val_every': 50,
        'start_run': 0,
        'train_runs': 2000,
        'agent_config': agent_config,
        'model_root': args.modelroot,
        'data_root': args.dataroot,
        'best_cost_eta': 0.5,
    }

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="train-vrp",
    #     name=experiment_config['experiment_name'],
    #     # track hyperparameters and run metadata
    #     config=experiment_config
    # )

    run_experiment(experiment_config, wandb_module=None, multiproc=False) # wandb)

    # wandb.finish()