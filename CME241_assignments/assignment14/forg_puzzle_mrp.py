from dataclasses import dataclass
from typing import Mapping, Dict, Tuple, List
from rl.distribution import Categorical, FiniteDistribution
from rl.markov_process import FiniteMarkovRewardProcess, NonTerminal
from rl.markov_process import MarkovRewardProcess
from rl.distribution import Categorical, Distribution
from rl.dynamic_programming import evaluate_mrp_result
from scipy.stats import poisson
from rl.distribution import SampledDistribution, Categorical, \
    FiniteDistribution
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class LilypadState:
    current_lilypad: int


class FrogPuzzleMRPFinite(FiniteMarkovRewardProcess[LilypadState]):

    def __init__(
        self,
        lilypad_num: int,
    ):
        self.lilypad_num: int = lilypad_num
        super().__init__(self.get_transition_reward_map())

    def get_transition_reward_map(self) -> \
            Mapping[LilypadState, FiniteDistribution[Tuple[LilypadState, float]]]:
        d: Dict[LilypadState, Categorical[Tuple[LilypadState, float]]] = {}
        for lilypad in range(self.lilypad_num - 1):
            state_reward_probs_map: Mapping[Tuple[LilypadState, float], float] = {}
            for j in range(self.lilypad_num-1-lilypad):
                state_reward_probs_map[(LilypadState(lilypad + j + 2), -1.0)] = 1/self.lilypad_num-1-lilypad
            d[LilypadState(lilypad+1)] = Categorical(state_reward_probs_map)
        print(d)
        return d

    def start_state_dist(self) -> Categorical[NonTerminal[LilypadState]]:
        start_dist = {}
        for i in range(self.lilypad_num):
            if i == 0:
                start_dist[NonTerminal(LilypadState(i+1))] = 1
            else:
                if i != self.lilypad_num-1:
                    start_dist[NonTerminal(LilypadState(i+1))] = 0
        return Categorical(start_dist)


if __name__ == '__main__':

    lilypad_number = 10
    simulation_times = 30000
    user_gamma = 1
    
    si_mrp = FrogPuzzleMRPFinite(lilypad_number)
    
    
    print("NonTerminal States")
    print("------------------")
    print(si_mrp.non_terminal_states)
    print()
    
    print("Transition Reward Map")
    print("------------------")
    print(si_mrp.transition_reward_map)
    print()

    print("Start State Distribution")
    print("------------------------")
    print(si_mrp.start_state_dist())
    print()

    # sum_frog_jumps = 0
    
    # for i in range(simulation_times):
    #     for k in list(si_mrp.simulate_reward(si_mrp.start_state_dist())):
    #         print(k.reward)
    #         sum_frog_jumps += k.reward
    
    # print("Total Frog Jumps")
    # print("------------------------")
    # print(sum_frog_jumps)
    # print()

    # expected_frog_jumps = 0
    # expected_frog_jumps = sum_frog_jumps/simulation_times
    
    # print("Expected Frog Jumps")
    # print("------------------------")
    # print(abs(expected_frog_jumps))
    # print()
    
    
    from rl.markov_process import FiniteMarkovProcess
    print("Transition Map")
    print("--------------")
    print(FiniteMarkovProcess(
        {s.state: Categorical({s1.state: p for s1, p in v.table().items()})
         for s, v in si_mrp.transition_map.items()}
    ))

  
    print("Reward Function")
    print("---------------")
    si_mrp.display_reward_function()
    print()

    print("Value Function")
    print("--------------")
    si_mrp.display_value_function(gamma=user_gamma)
    print()
    
    
    print("NonTerminal States")
    print("------------------")
    print(si_mrp.non_terminal_states)
    print()
    
    
    print("Evaulate MRP Result")
    print("-------------------")
    print(evaluate_mrp_result(si_mrp, gamma=user_gamma))
    print()

    print("Start State Distribution")
    print("------------------")
    print(si_mrp.start_state_dist())
    print()
    
    # use simulate_reward to  generate transitions (itertools. isslice)
    print("Transitions")
    print("-----------")
    simluate_reward = si_mrp.simulate_reward(si_mrp.start_state_dist())
    print(list(simluate_reward))
    print()
    

    # use simulate_reward to generate episodes
    print("Episodes")
    print("--------")
    episodes = []
    for i in range(500):
        episodes.append(list(si_mrp.simulate_reward(si_mrp.start_state_dist())))
    print(episodes)  
    print()