import os.path
import argparse

import cv2
import numpy as np
from tqdm import tqdm

from deepls.TSP2OptEnv import TSP2OptMultiEnv
from deepls.agent import AverageStateRewardBaselineAgent
from deepls.solver import greedy_postproc


TSP_SIZE_TO_TEST_DATA_F = {
    10: 'tsp10_test_concorde.txt',
    20: 'tsp20_test_concorde.txt',
    30: 'tsp30_test_concorde.txt',
    50: 'tsp50_test_concorde.txt',
    100: 'tsp100_test_concorde.txt',
}

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


def moving_average(arr, c=100):
    return np.convolve(arr, np.ones(shape=(c,))/c, mode='valid')


def run_experiment(
    experiment_config
):
    data_root = experiment_config['data_root']

    device = experiment_config['device']
    problem_sz = experiment_config['problem_sz']
    model_ckpt = experiment_config.get('model_ckpt')
    num_samples_per_batch = experiment_config['num_samples_per_batch']
    num_instance_eval = experiment_config['num_instance_eval']
    eval_batch_size = min(
        experiment_config['eval_batch_size'],
        num_samples_per_batch
    )
    num_greedy_postproc_steps = experiment_config['num_greedy_postproc_steps']

    env = TSP2OptMultiEnv(
        max_num_steps=problem_sz,
        num_nodes=problem_sz,
        data_f=f'{data_root}/{TSP_SIZE_TO_TEST_DATA_F[problem_sz]}',
        num_samples_per_batch=num_samples_per_batch,
        same_instance_per_batch=True,
        shuffle_data=True,
        ret_log_tour_len=False
    )
    env.reset()

    agent = AverageStateRewardBaselineAgent()
    if model_ckpt is not None:
        agent.load(model_ckpt, init_config=True, device=device)

    all_opt_gaps = []
    all_opt_gaps_pre = []
    all_tour_lens = []
    all_opts = []

    agent.set_eval()
    agent.set_greedy(False)

    for _ in tqdm(range(num_instance_eval)):

        tour_lens = []
        opts = []
        opt_gaps_pre = []
        opt_gaps = []

        pbar = tqdm(range(num_samples_per_batch // eval_batch_size))
        env.reset(fetch_next=True)
        for episode in pbar:
            env.reset(fetch_next=False)
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
                    opt_gaps_pre.extend(
                        [(state[1].tour_len / state[0].opt_tour_len) - 1. for state in states]
                    )
                    post_proc_states = greedy_postproc(states, num_postproc_steps=num_greedy_postproc_steps)
                    tour_lens.extend([state[1].tour_len for state in post_proc_states])
                    opts.extend([state[0].opt_tour_len for state in post_proc_states])
                    opt_gaps.extend(
                        [(state[1].tour_len / state[0].opt_tour_len) - 1. for state in post_proc_states]
                    )
                    break
                else:
                    actions = agent.agent_step(rewards, states)
        all_opt_gaps.append(opt_gaps)
        all_opt_gaps_pre.append(opt_gaps_pre)
        all_tour_lens.append(tour_lens)
        all_opts.append(opts)

    best_opt_gaps = np.min(all_opt_gaps, axis=1)
    best_opt_gaps_pre = np.min(all_opt_gaps_pre, axis=1)
    print(
        f"average best optimiality gap (with postproc) = {np.mean(best_opt_gaps):.3f}\n"
        f"average best optimiality gap (w/o postproc) =  {np.mean(best_opt_gaps_pre):.3f}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        "-d",
        default=os.path.join(os.path.dirname(__file__), 'data', 'tsp-data'),
        help="location of data files"
    )
    parser.add_argument(
        "--ckptpath",
        "-c"
    )
    args = parser.parse_args()

    experiment_config = {
        'problem_sz': 20,
        'num_samples_per_batch': 1,
        'num_instance_eval': 100,
        'eval_batch_size': 12,
        'num_greedy_postproc_steps': 10,
        'data_root': args.dataroot,
        'model_ckpt': args.ckptpath,
        'device': 'cuda'
    }

    run_experiment(experiment_config)