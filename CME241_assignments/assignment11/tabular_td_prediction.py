import itertools
import numpy as np

from typing import Callable, Iterable, Iterator, TypeVar, Set, Sequence, Tuple, Mapping, List
import math

import rl.markov_process as mp
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_decision_process import TransitionStep, NonTerminal
from operator import itemgetter
from rl.approximate_dynamic_programming import ValueFunctionApprox, \
    QValueFunctionApprox, NTStateDistribution, extended_vf
from rl.distribution import Categorical
from forg_puzzle_mrp import FrogPuzzleMRPFinite, LilypadState
from rl.function_approx import LinearFunctionApprox, Weights
import rl.iterate as iterate
import rl.markov_process as mp
from rl.monte_carlo import greedy_policy_from_qvf
from rl.policy import Policy, DeterministicPolicy
from rl.experience_replay import ExperienceReplayMemory
from rl.function_approx import AdamGradient
from rl.dynamic_programming import evaluate_mrp_result

S = TypeVar('S')

def tabular_td_prediction(
    non_terminal_set: List[NonTerminal[S]],
    transitions: Iterable[mp.TransitionStep[S]],
    γ: float
) -> Mapping[NonTerminal[S], float]:

    values_map: Mapping[NonTerminal[Ｓ], float] = {}
    counts_map: Mapping[NonTerminal[Ｓ], int] = {}
    count_to_weight_func: Callable[[int], float] = lambda n: 1/n
    
    for non_terminal_state in non_terminal_set:
        values_map[non_terminal_state] = 0.0
        counts_map[non_terminal_state] = 0
    
    for trans in transitions:
        if trans.state in values_map:
            counts_map[trans.state] += 1
            if trans.next_state not in values_map:
                values_map[trans.state] += count_to_weight_func(counts_map[trans.state]) * (trans.reward - values_map[trans.state])
            else:
                values_map[trans.state] += count_to_weight_func(counts_map[trans.state]) * (trans.reward + γ * values_map[trans.next_state]  - values_map[trans.state])        
                 
    return values_map
    
    

def td_prediction(
        transitions: Iterable[mp.TransitionStep[S]],
        approx_0: ValueFunctionApprox[S],
        γ: float
) -> Iterator[ValueFunctionApprox[S]]:

    def step(
            v: ValueFunctionApprox[S],
            transition: mp.TransitionStep[S]
    ) -> ValueFunctionApprox[S]:
        return v.update([(
            transition.state,
            transition.reward + γ * extended_vf(v, transition.next_state)
        )])
    return iterate.accumulate(transitions, step, initial=approx_0)
    
    
    
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
    
    print("Tabular TD")
    print("-------------------")
    print(tabular_td_prediction(si_mrp.non_terminal_states,transitions, user_gamma))
    tabular_td_result = tabular_td_prediction(si_mrp.non_terminal_states,transitions, user_gamma)
    print()
    
    ffs: Sequence[Callable[[NonTerminal[LilypadState]], float]] = \
    [(lambda x, s=s: float(x.state == s.state)) for s in si_mrp.non_terminal_states]
    
    mc_ag: AdamGradient = AdamGradient(
    learning_rate=0.05,
    decay1=0.9,
    decay2=0.999)

    td_func_approx: LinearFunctionApprox[NonTerminal[LilypadState]] = \
    LinearFunctionApprox.create(
        feature_functions=ffs,
        adam_gradient=mc_ag
    )
    
    print("TD Function Approximation")
    print("-------------------------")
    print(td_prediction(transitions, td_func_approx, user_gamma))
    td_func_result = list(td_prediction(transitions, td_func_approx, user_gamma))[-1].weights.weights
    print()
    
    td_dict = {}
    for i in range(len(si_mrp.non_terminal_states)):
        td_dict[si_mrp.non_terminal_states[i]] =  td_func_result[i]
    print(td_dict)
    print() 
    
    convergence = 0
    
    for ns in tabular_td_result:
        convergence += (tabular_td_result[ns] - td_dict[ns]) ** 2
    
    convergence = math.sqrt(convergence)
    print("Convergence:", convergence)