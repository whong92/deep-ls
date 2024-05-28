from deepls.vrp_gcn_model import AverageStateRewardBaselineAgentVRP, VRP_STANDARD_PROBLEM_CONF
from deepls.VRPState import VRPMultiRandomEnv, plot_state, VRPMultiFileEnv, VRPState, VRPEnvBase, VRPReward, VRPMultiFileEnvSingleProc, VectorizedState
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from deepls.vrp_greedy_solver import greedy_sample, VRPNbH
import json

from typing import List, Tuple

episodes = 10
hidden_dim = 128

N = 50
num_steps = 100
max_tour_demand = VRP_STANDARD_PROBLEM_CONF[N]['capacity']

agent_config = {
    'replay_buffer_size': 1,
    'batch_sz': 64,
    'minibatch_sz': 32,
    'policy_optimize_every': 2,
    'use_ppo_update': True,
    'model': {
        "node_dim": 2,
        "voc_edges_in": 3,
        "hidden_dim": hidden_dim,
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

workdir = '/home/ong/personal/deep-ls-tsp'

agent = AverageStateRewardBaselineAgentVRP()
agent.agent_init(agent_config)
agent.load(f'{workdir}/model/vrp-50-nodes-lr-2e-6-beta-2e-3-longer-delta-cost-singleton-init-from-scratch/model-04000-val-0.071.ckpt', init_config=False)
agent.set_eval()
# agent.set_train()

# envs = VRPMultiFileEnvSingleProc(
#     data_f=f'{workdir}/data/vrp-data/size-50/vrp_data_with_results.pkl',
#     num_nodes=N,
#     max_num_steps=num_steps,
#     max_tour_demand=max_tour_demand,
#     num_samples_per_instance=12,
#     num_instance_per_batch=1,
#     reward_mode=VRPReward.DELTA_COST
# )
envs = VRPMultiFileEnv(
    data_f=f'{workdir}/data/vrp-data/size-50/vrp_data_with_results.pkl',
    num_nodes=N,
    max_num_steps=num_steps,
    max_tour_demand=max_tour_demand,
    num_samples_per_instance=12,
    num_instance_per_batch=1,
    reward_mode=VRPReward.DELTA_COST,
    vectorize_state=True
)
pbar = tqdm(range(episodes))
opt_gaps = 0.

state_opts_all = []
best_opts_all = []

for episode in pbar:

    state_opts = []
    best_opts = []
    step = 0
    # env.set_instance_as_state(instance, id=episode, max_num_steps=num_steps)
    states: List[VectorizedState] = envs.reset(fetch_next=True)
    actions = agent.agent_start(states, envs)
    # init_cost = states[0][1].get_cost(exclude_depot=False)
    init_cost = states[0].states_cost[0]
    # opt_cost = states[0][0].opt_tour_dist
    opt_cost = states[0].states_opt_cost[0]

    # state_opts.append([state[0].get_cost(exclude_depot=False) / opt_cost - 1. for state in states])
    # best_opts.append([state[1].get_cost(exclude_depot=False) / opt_cost - 1. for state in states])
    state_opts.append([state.states_cost[0] / opt_cost - 1. for state in states])
    best_opts.append([state.best_states_cost[0] / opt_cost - 1. for state in states])

    # plot_state(states[0][0], f'{workdir}/dump/episode_{episode:03d}_step_{step:03d}.jpg')
    while True:
        step += 1
        states, rewards, dones = envs.step(actions)
        # plot_state(states[0][0], f'{workdir}/dump/episode_{episode:03d}_step_{step:03d}.jpg')
        # state_opts.append([state[0].get_cost(exclude_depot=False) / opt_cost - 1. for state in states])
        # best_opts.append([state[1].get_cost(exclude_depot=False) / opt_cost - 1. for state in states])
        state_opts.append([state.states_cost[0] / opt_cost - 1. for state in states])
        best_opts.append([state.best_states_cost[0] / opt_cost - 1. for state in states])
        done = dones[0]
        if done:
            print(agent.agent_end(rewards))
            break
        else:
            actions = agent.agent_step(
                rewards,
                states,
                envs
            )

    state_opts_all.append(state_opts)
    best_opts_all.append(best_opts)

print(opt_gaps / episodes)


opts_all = {
    'state_opts_all': state_opts_all,
    'best_opts_all': best_opts_all,
}
with open("viz_eval_opts_all_beta_2e-3_long_delta_vectorized_impl.json", "w") as fp:
    json.dump(opts_all, fp)