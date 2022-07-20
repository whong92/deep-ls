import copy
from typing import List, Tuple

from deepls.TSP2OptEnv import TSP2OptState, TSP2OptEnvBase
from deepls.agent import GreedyAgent


def greedy_postproc(states: List[Tuple[TSP2OptState, TSP2OptState]], num_postproc_steps: int):
    final_states = []
    for state in states:
        env = TSP2OptEnvBase(max_num_steps=num_postproc_steps)
        env.set_state(state[1])
        gagent = GreedyAgent()
        state = env.get_state()
        action = gagent.agent_start([state])[0]
        while True:
            # Take a random action
            state, reward, done = env.step(action)
            if done:
                gagent.agent_end([reward])
                final_states.append(copy.deepcopy(state))
                break
            else:
                action = gagent.agent_step([reward], [state])[0]
    return final_states
