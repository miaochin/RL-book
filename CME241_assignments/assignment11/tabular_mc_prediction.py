
from typing import Iterable, Iterator, TypeVar, Callable, Mapping, List, Set, Sequence, Tuple
from rl.markov_decision_process import MarkovDecisionProcess, Policy, \
    TransitionStep, NonTerminal
from rl.policy import DeterministicPolicy, RandomPolicy, UniformPolicy, Policy
import rl.markov_process as mp
from rl.returns import returns
from operator import itemgetter
from rl.iterate import last
from forg_puzzle_mrp import FrogPuzzleMRPFinite, LilypadState
from rl.markov_process import FiniteMarkovProcess
from rl.approximate_dynamic_programming import ValueFunctionApprox, \
    QValueFunctionApprox, NTStateDistribution, extended_vf
from rl.distribution import Categorical
from rl.function_approx import LinearFunctionApprox, Weights
import rl.iterate as iterate
from rl.markov_decision_process import MarkovDecisionProcess
from rl.monte_carlo import greedy_policy_from_qvf
from rl.experience_replay import ExperienceReplayMemory
from rl.dynamic_programming import evaluate_mrp_result
from rl.function_approx import AdamGradient
from rl.function_approx import LinearFunctionApprox
from rl.approximate_dynamic_programming import ValueFunctionApprox

import itertools
import numpy as np
import math

S = TypeVar('S')

def tabular_mc_prediction(
    non_terminal_set: List[NonTerminal[S]],
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    γ: float,
    episode_length_tolerance: float = 1e-6
) -> Mapping[NonTerminal[S], float]:

    
    values_map: Mapping[NonTerminal[Ｓ], float] = {}
    counts_map: Mapping[NonTerminal[Ｓ], int] = {}
    count_to_weight_func: Callable[[int], float] = lambda n: 1/n
    
    for non_terminal_state in non_terminal_set:
        values_map[non_terminal_state] = 0.0
        counts_map[non_terminal_state] = 0
        
    
    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ, episode_length_tolerance) for trace in traces)

    for episode in episodes:
        for step in episode:
            if step.state in values_map:
                counts_map[step.state] += 1
                values_map[step.state] = values_map[step.state] + count_to_weight_func(counts_map[step.state]) *\
                    (step.return_-values_map[step.state])
                    
    return values_map
    
    
def mc_prediction(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    approx_0: ValueFunctionApprox[S],
    γ: float,
    episode_length_tolerance: float = 1e-6
) -> Iterator[ValueFunctionApprox[S]]:
    
    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ, episode_length_tolerance) for trace in traces)
    f = approx_0
    yield f

    for episode in episodes:
        f = last(f.iterate_updates(
            [(step.state, step.return_)] for step in episode
        ))
        yield f

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
    

    # use simulate_reward to generate episodes
    print("Episodes")
    print("--------")
    episodes = []
    for i in range(1000):
        episodes.append(list(si_mrp.simulate_reward(si_mrp.start_state_dist())))
    #print(episodes)  
    print()
    
    print("Value Function")
    print("--------------")
    si_mrp.display_value_function(gamma=user_gamma)
    print()
    
    print("Evaulate MRP Result")
    print("-------------------")
    print(evaluate_mrp_result(si_mrp, gamma=user_gamma))
    mrp_result = evaluate_mrp_result(si_mrp, gamma=user_gamma)
    print()
    
    print("Tabular Monte Carlo")
    print("-------------------")
    tabular_mc_result = tabular_mc_prediction(si_mrp.non_terminal_states,episodes, user_gamma)
    print(tabular_mc_prediction(si_mrp.non_terminal_states,episodes, user_gamma))
    print()
    
    ffs: Sequence[Callable[[NonTerminal[LilypadState]], float]] = \
    [(lambda x, s=s: float(x.state == s.state)) for s in si_mrp.non_terminal_states]
    
    mc_ag: AdamGradient = AdamGradient(
    learning_rate=0.05,
    decay1=0.9,
    decay2=0.999)

    mc_func_approx: LinearFunctionApprox[NonTerminal[LilypadState]] = \
    LinearFunctionApprox.create(
        feature_functions=ffs,
        adam_gradient=mc_ag
    )
    
    print("Monte Carlo Function Approximation")
    print("-------------------")
    print((mc_prediction(episodes, mc_func_approx, 1)))
    mc_func_result = list((mc_prediction(episodes, mc_func_approx, 1)))[-1].weights.weights
    print()
    
    mc_dict = {}
    for i in range(len(si_mrp.non_terminal_states)):
        mc_dict[si_mrp.non_terminal_states[i]] =  mc_func_result[i]
    print(mc_dict)
    print() 
    
    convergence = 0
    for ns in tabular_mc_result:
        convergence += (tabular_mc_result[ns] - mc_dict[ns]) ** 2
    convergence = math.sqrt(convergence)
    print("Convergence:", convergence)
    
    convergence = 0
    for ns in tabular_mc_result:
        convergence += (tabular_mc_result[ns] - mrp_result[ns]) ** 2
    convergence = math.sqrt(convergence)
    print("Convergence:", convergence)
    
    convergence = 0
    for ns in tabular_mc_result:
        convergence += (mrp_result[ns] - mc_dict[ns]) ** 2
    convergence = math.sqrt(convergence)
    print("Convergence:", convergence)
    
    