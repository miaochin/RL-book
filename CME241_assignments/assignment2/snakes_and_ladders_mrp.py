from dataclasses import dataclass
from typing import Mapping, Dict, Tuple, List

from pkg_resources import Distribution
from rl.distribution import Categorical, FiniteDistribution
from rl.markov_process import FiniteMarkovRewardProcess, NonTerminal

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PositionState:
    current_pos: int


class SnakesAndLaddersMRPFinite(FiniteMarkovRewardProcess[PositionState]):

    def __init__(
        self,
        grid_num: int,
        snake_ladder: List[Tuple[int, int]]
    ):
        self.grid_num: int = grid_num
        self.snake_ladder = snake_ladder
        super().__init__(self.get_transition_reward_map())

    def get_transition_reward_map(self) -> \
            Mapping[PositionState, FiniteDistribution[Tuple[PositionState, float]]]:
        d: Dict[PositionState, Categorical[Tuple[PositionState, float]]] = {}
        for position in range(self.grid_num - 1):
            state_reward_probs_map: Mapping[Tuple[PositionState, float], float] = {}
            i = 0
            while i < 6:
                if i + position + 2 < self.grid_num:
                    state_reward_probs_map[(PositionState(position + i + 2), -1.0)] = 1/6
                    i += 1
                else:
                    break
            state_reward_probs_map[(PositionState(self.grid_num), -1.0)] = 1 - 1/6 * i
            print(state_reward_probs_map, 'jerereererw')
            d[PositionState(position+1)] = Categorical(state_reward_probs_map)
        for pair in self.snake_ladder:
            d[PositionState(pair[0])] = Categorical({(PositionState(pair[1]),0.0): 1.0})
        print(d)
        return d

    def start_state_dist(self) -> Categorical[NonTerminal[PositionState]]:
        start_dist = {}
        for i in range(self.grid_num):
            if i == 0:
                start_dist[NonTerminal(PositionState(i+1))] = 1
            else:
                if i != self.grid_num-1:
                    start_dist[NonTerminal(PositionState(i+1))] = 0
        return Categorical(start_dist)


if __name__ == '__main__':

    grid_number = 100
    simulation_times = 80000
    
    snake_ladder = [(1,38), (4, 14), (9, 31), (28, 84), (80, 100), (71, 91), (93, 73), (95, 75), (87, 24), (47, 26), (17,6)]


    si_mrp = SnakesAndLaddersMRPFinite(grid_number, snake_ladder)
    
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
    sum_dice_rolls = 0
    
    for i in range(simulation_times):
        for k in list(si_mrp.simulate_reward(si_mrp.start_state_dist())):
            sum_dice_rolls += k.reward
            
    print("Total Number of Dice Rolls")
    print("------------------------")
    print(sum_dice_rolls)
    print()

    expected_dice_rolls = 0
    expected_dice_rolls = sum_dice_rolls/simulation_times
    
    print("Expected Number of Dice Rolls")
    print("------------------------")
    print(abs(expected_dice_rolls))
    print()
            
            




