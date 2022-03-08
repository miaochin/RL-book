from dataclasses import dataclass
from typing import List, Mapping, Dict, Tuple
from rl.distribution import Categorical, FiniteDistribution
from rl.markov_process import FiniteMarkovProcess, NonTerminal

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PositionState:
    current_pos: int


class SnakesAndLaddersMPFinite(FiniteMarkovProcess[PositionState]):

    def __init__(
        self,
        grid_num: int,
        snake_ladder: List[Tuple[int, int]]
    ):
        self.grid_num: int = grid_num
        self.snake_ladder = snake_ladder
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> \
            Mapping[PositionState, FiniteDistribution[PositionState]]:
        d: Dict[PositionState, Categorical[PositionState]] = {}
        for position in range(self.grid_num - 1):
            state_probs_map: Mapping[PositionState, float] = {}
            i = 0
            while i < 6:
                if i + position + 2 < self.grid_num:
                    state_probs_map[PositionState(position + i + 2)] = 1/6
                    i += 1
                else:
                    break
            state_probs_map[PositionState(self.grid_num)] = 1 - 1/6 * i
            print(state_probs_map)
            d[PositionState(position+1)] = Categorical(state_probs_map)
        for pair in self.snake_ladder:
            d[PositionState(pair[0])] = Categorical({PositionState(pair[1]): 1.0})
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


    si_mp = SnakesAndLaddersMPFinite(grid_number, snake_ladder)
    
    
    print("NonTerminal States")
    print("------------------")
    print(si_mp.non_terminal_states)
    print()
    
    print("Transition Map")
    print("------------------")
    print(si_mp.transition_map)
    print()
    
    
    # print("Start State Distribution")
    # print("------------------------")
    # print(si_mp.start_state_dist())
    # print()

    time_step_dict = {}

    for i in range(simulation_times):
        time_steps = len(list(si_mp.simulate(si_mp.start_state_dist())))
        if time_steps not in time_step_dict:
            time_step_dict[time_steps] = 1
        else:
            time_step_dict[time_steps] += 1

    prob_dist_time_step = Categorical(time_step_dict)
    
    print("Time Steps Probability Distribution")
    print("------------------------------------")
    print(prob_dist_time_step)
    print()

    x_data = []
    y_data = []

    for i in prob_dist_time_step:
        x_data.append(i[0])
    x_data.sort()
    
    for i in x_data:
        y_data.append(prob_dist_time_step.probability(i))
        
    print(x_data)
    print(y_data)

    x = np.array(x_data)
    y = np.array(y_data)

    plt.plot(x, y, linestyle='--')

    # Add Title

    plt.title("Probability Distribution of Time Steps to Finish the Gamme")

    # Add Axes Labels

    plt.xlabel("Time Steps to Finish the Game")
    plt.ylabel("Probability")

    # Display

    plt.show()
    