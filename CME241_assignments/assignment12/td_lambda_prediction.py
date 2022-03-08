from typing import Iterable, Iterator, TypeVar, List, Sequence, Mapping, Callable
from rl.function_approx import Gradient
import rl.markov_process as mp
from rl.markov_decision_process import NonTerminal
import numpy as np
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.approximate_dynamic_programming import extended_vf
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
import math
import matplotlib.pyplot as plt

S = TypeVar('S')


def tabular_td_lambda(
    non_terminal_set: List[NonTerminal[S]],
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    γ: float,
    alpha: float,
    lambd: float,
) -> Mapping[NonTerminal[S], float]:

    values_map: Mapping[NonTerminal[S], float] = {s: 0 for s in non_terminal_set}
    
    for trace in traces:
        el_tr: Mapping[NonTerminal[S], float] = {s:0 for s in non_terminal_set}
        for step in trace:
            for s in el_tr:
                if step.state == s:
                    el_tr[s] = γ * lambd * el_tr[s] + 1
                else:
                    el_tr[s] = γ * lambd * el_tr[s]
                    
            if step.state in values_map:
                if step.next_state not in values_map:
                    values_map[step.state] += alpha * (step.reward - values_map[step.state]) * el_tr[step.state]
                else:
                    values_map[step.state] += alpha * (step.reward + γ * values_map[step.next_state]  - values_map[step.state]) * \
                        el_tr[step.state]
    return values_map
    


def td_lambda_prediction(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        approx_0: ValueFunctionApprox[S],
        γ: float,
        lambd: float
) ->  Iterator[ValueFunctionApprox[S]]:

    func_approx: ValueFunctionApprox[S] = approx_0
    yield func_approx

    for trace in traces:
        el_tr: Gradient[ValueFunctionApprox[S]] = Gradient(func_approx).zero()
        for step in trace:
            x: NonTerminal[S] = step.state
            y: float = step.reward + γ * \
                extended_vf(func_approx, step.next_state)
            el_tr = el_tr * (γ * lambd) + func_approx.objective_gradient(
                xy_vals_seq=[(x, y)],
                obj_deriv_out_fun=lambda x1, y1: np.ones(len(x1))
            )
            func_approx = func_approx.update_with_gradient(
                el_tr * (func_approx(x) - y)
            )
            yield func_approx
            
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
    mrp_result = evaluate_mrp_result(si_mrp, gamma=user_gamma)
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

    
    print("Tabular TD Lambda")
    print("-------------------")
    print(tabular_td_lambda(si_mrp.non_terminal_states,episodes, user_gamma, 0.05, 0.5))
    tabular_td_lambda_result = tabular_td_lambda(si_mrp.non_terminal_states,episodes, user_gamma, 0.05, 0.5)
    print()
    
    ffs: Sequence[Callable[[NonTerminal[LilypadState]], float]] = \
    [(lambda x, s=s: float(x.state == s.state)) for s in si_mrp.non_terminal_states]
    
    mc_ag: AdamGradient = AdamGradient(
    learning_rate=0.05,
    decay1=0.9,
    decay2=0.999)

    td_lambda_func_approx: LinearFunctionApprox[NonTerminal[LilypadState]] = \
    LinearFunctionApprox.create(
        feature_functions=ffs,
        adam_gradient=mc_ag
    )
    
    print("TD Lambda Function Approximation")
    print("-------------------")
    print((td_lambda_prediction(episodes, td_lambda_func_approx, user_gamma, 0.5)))
    td_lambda_func_result = list((td_lambda_prediction(episodes, td_lambda_func_approx, 0.7, 0.5)))[-1].weights.weights
    print()
    
    def td_dict(dict1): 
        td_lambda_dict = {}
        for i in range(len(si_mrp.non_terminal_states)):
            td_lambda_dict[si_mrp.non_terminal_states[i]] =  dict1[i]
        print(td_lambda_dict)
        print()      
        return td_lambda_dict
    
    
    def convergence_calc(dict1, dict2):
        convergence = 0
        for ns in dict1:
            convergence += (dict1[ns] - dict2[ns]) ** 2
        convergence = math.sqrt(convergence)
        print("Convergence:", convergence)
        
        return convergence
        
    convergence = []
    for lamb in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(lamb)
        result = convergence_calc(mrp_result,td_dict(list((td_lambda_prediction(episodes, td_lambda_func_approx, user_gamma, lamb)))[-1].weights.weights))
        convergence.append(result)
        
    print(convergence)
    
    

    x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y = np.array(convergence)

    plt.plot(x, y, linestyle='--')

    # Add Axes Labels

    plt.xlabel("Lambda")
    plt.ylabel("Distance")

    # Display

    plt.show()
    
    