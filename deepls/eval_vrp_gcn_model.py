from deepls.vrp_gcn_model import AverageStateRewardBaselineAgentVRP, VRP_STANDARD_PROBLEM_CONF
from deepls.VRPState import VRPMultiRandomEnv, plot_state, VRPMultiFileEnv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":

    episodes = 10
    hidden_dim = 128

    N = 20
    num_steps = 40
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
    agent.load(f'{workdir}/model/vrp-20-nodes-chunked-episodes/model-00500-val-0.028.ckpt', init_config=False)
    agent.set_eval()

    envs = VRPMultiFileEnv(
        data_f=f'{workdir}/data/vrp-data/size-20/vrp_data_with_results_val.pkl',
        num_nodes=N,
        max_num_steps=num_steps,
        max_tour_demand=max_tour_demand,
        num_samples_per_instance=1,
        num_instance_per_batch=1
    )
    pbar = tqdm(range(episodes))

    for episode in pbar:

        step = 0
        # env.set_instance_as_state(instance, id=episode, max_num_steps=num_steps)
        states = envs.reset(fetch_next=True)
        actions = agent.agent_start(states)
        init_cost = states[0][1].get_cost(exclude_depot=False)
        opt_cost = states[0][0].opt_tour_dist
        plot_state(states[0][0], f'{workdir}/dump/episode_{episode:03d}_step_{step:03d}.jpg')
        while True:
            step += 1
            states, rewards, dones = envs.step(actions)
            plot_state(states[0][0], f'{workdir}/dump/episode_{episode:03d}_step_{step:03d}.jpg')
            done = dones[0]
            if done:
                agent.agent_end(rewards)
                break
            else:
                actions = agent.agent_step(
                    rewards,
                    states
                )
        best_state_cost = states[0][1].get_cost(exclude_depot=False)
        print(f'opt_gap = {best_state_cost:.3f} / {opt_cost:.3f} = {(1. - best_state_cost/opt_cost):.3f}')
