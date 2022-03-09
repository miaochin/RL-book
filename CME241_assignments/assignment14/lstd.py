

from operator import itemgetter
import itertools
from typing import Callable, Iterable, Iterator, TypeVar, Set, Sequence, Tuple
from unicodedata import name

import numpy as np
from forg_puzzle_mrp import LilypadState, FrogPuzzleMRPFinite
from rl.approximate_dynamic_programming import ValueFunctionApprox, \
    QValueFunctionApprox, NTStateDistribution, extended_vf
from rl.distribution import Categorical
from rl.function_approx import LinearFunctionApprox, Weights
import rl.iterate as iterate
import rl.markov_process as mp
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_decision_process import TransitionStep, NonTerminal
from rl.monte_carlo import greedy_policy_from_qvf
from rl.policy import Policy, DeterministicPolicy
from rl.experience_replay import ExperienceReplayMemory
from rl.td import least_squares_policy_iteration
from rl.td import least_squares_td
from rl.function_approx import AdamGradient
from rl.dynamic_programming import evaluate_mrp_result



if __name__ == '__main__':

    lilypad_number = 10
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
    
    # use simulate_reward to generate transitions
    print("Transitions")
    print("--------")
    transitions = []
    for i in range(1000):
        for trans in list(si_mrp.simulate_reward(si_mrp.start_state_dist())):
            transitions.append(trans)    
    # print(transitions)
    print()
    
    print("Value Function")
    print("--------------")
    si_mrp.display_value_function(gamma=user_gamma)
    print()
    
    print("Evaulate MRP Result")
    print("-------------------")
    print(evaluate_mrp_result(si_mrp, gamma=user_gamma))
    print()
    
    
    ffs: Sequence[Callable[[NonTerminal[LilypadState]], float]] = \
    [(lambda x, s=s: float(x.state == s.state)) for s in si_mrp.non_terminal_states]
    

    
    lstd = least_squares_td(
        transitions,
        feature_functions=ffs,
        γ=1 ,
        ε=0.2 
    )
    
    print("LSTD Function Approximation")
    print("-------------------------")
    lstd_result = lstd.weights.weights
    print(lstd_result)
    print()
    
    lstd_dict = {}
    for i in range(len(si_mrp.non_terminal_states)):
        lstd_dict[si_mrp.non_terminal_states[i]] =  lstd_result[i]
    print(lstd_dict)
    print() 
