from deepls.vrp_gcn_model import AverageStateRewardBaselineAgentVRP, VRP_STANDARD_PROBLEM_CONF
from deepls.VRPState import VRPMultiRandomEnv, plot_state, VRPMultiFileEnv, VRPState, VRPEnvBase
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from deepls.vrp_greedy_solver import greedy_sample, VRPNbH

from typing import List, Tuple

if __name__=="__main__":

    episodes = 30
    hidden_dim = 128

    N = 50
    num_steps = 200
    max_tour_demand = VRP_STANDARD_PROBLEM_CONF[N]['capacity']

    agent_config = {
        'replay_buffer_size': 10,
        'batch_sz': 64,
        'minibatch_sz': 32,
        'policy_optimize_every': 2,
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
        'device': 'cpu'
    }

    workdir = '/home/ong/personal/deep-ls-tsp'

    agent = AverageStateRewardBaselineAgentVRP()
    agent.agent_init(agent_config)
    agent.load(f'{workdir}/model/vrp-50-nodes-chunked-episodes-small-lr-fixed/model-02500-val-0.089.ckpt', init_config=False)
    agent.set_eval()

    envs = VRPMultiFileEnv(
        data_f=f'{workdir}/data/vrp-data/size-50/vrp_data_with_results.pkl',
        num_nodes=N,
        max_num_steps=num_steps,
        max_tour_demand=max_tour_demand,
        num_samples_per_instance=1,
        num_instance_per_batch=1
    )
    pbar = tqdm(range(episodes))
    opt_gaps = 0.

    for episode in pbar:

        step = 0
        # env.set_instance_as_state(instance, id=episode, max_num_steps=num_steps)
        states: List[Tuple[VRPState, VRPState]] = envs.reset(fetch_next=True)
        actions = agent.agent_start(states)
        init_cost = states[0][1].get_cost(exclude_depot=False)
        opt_cost = states[0][0].opt_tour_dist
        # plot_state(states[0][0], f'{workdir}/dump/episode_{episode:03d}_step_{step:03d}.jpg')
        while True:
            step += 1
            states, rewards, dones = envs.step(actions)
            # plot_state(states[0][0], f'{workdir}/dump/episode_{episode:03d}_step_{step:03d}.jpg')
            done = dones[0]
            if done:
                agent.agent_end(rewards)
                break
            else:
                actions = agent.agent_step(
                    rewards,
                    states
                )

        num_greedy_steps = 20
        cur_instances = envs.get_instance()
        env_greedy = VRPEnvBase(max_num_steps=num_greedy_steps, max_tour_demand=max_tour_demand)
        state = states[0]
        env_greedy.set_instance_as_state(
            cur_instances[0],
            init_tour=state[1].all_tours_as_list(remove_last_depot=True, remove_first_depot=True)
        )
        _, moves = greedy_sample([VRPNbH(env_greedy.get_state()[0])])
        for _ in range(num_greedy_steps):
            if moves[0]['cost'] > 0:
                break
            state, reward, done = env_greedy.step({'move': moves[0], 'terminate': False})
            # plot_state(state[0], f'{workdir}/dump/episode_{episode:03d}_step_{step:03d}.jpg')
            if done:
                break
            else:
                _, moves = greedy_sample([VRPNbH(state[0])])
        best_state_cost = state[1].get_cost(exclude_depot=False)
        opt_gap = best_state_cost/opt_cost - 1.
        opt_gaps += opt_gap
        print(f'opt_gap = {best_state_cost:.3f} / {opt_cost:.3f} = {(best_state_cost/opt_cost - 1.):.3f}')
    print(opt_gaps / episodes)