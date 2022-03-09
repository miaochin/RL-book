from rl.td import q_learning
from rl.monte_carlo import epsilon_greedy_policy
from rl.markov_decision_process import MarkovDecisionProcess
from operator import itemgetter
import itertools
from typing import Callable, Iterable, Iterator, TypeVar, Set, Sequence, Tuple, Dict, Mapping
import numpy as np
from rl.approximate_dynamic_programming import ValueFunctionApprox, \
    QValueFunctionApprox, NTStateDistribution, extended_vf
from rl.distribution import Categorical
from rl.function_approx import LinearFunctionApprox, Weights, Tabular, learning_rate_schedule
import rl.iterate as iterate
import rl.markov_process as mp
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_decision_process import TransitionStep, NonTerminal
from rl.monte_carlo import greedy_policy_from_qvf
from rl.policy import Policy, DeterministicPolicy
from rl.experience_replay import ExperienceReplayMemory
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap, InventoryState
from rl.distribution import Choose
from rl.dynamic_programming import evaluate_mrp_result
from rl.dynamic_programming import policy_iteration_result
from rl.dynamic_programming import value_iteration_result
from pprint import pprint

capacity: int = 2
poisson_lambda: float = 1.0
holding_cost: float = 1.0
stockout_cost: float = 10.0

si_mdp: SimpleInventoryMDPCap = SimpleInventoryMDPCap(
    capacity=capacity,
    poisson_lambda=poisson_lambda,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost
)

gamma: float = 0.9
mc_episode_length_tol: float = 1e-5
num_episodes = 10000

epsilon_as_func_of_episodes: Callable[[int], float] = lambda k: k ** -0.5
q_learning_epsilon: float = 0.2

td_episode_length: int = 100
initial_learning_rate: float = 0.1
half_life: float = 10000.0
exponent: float = 1.0

initial_qvf_dict: Mapping[Tuple[NonTerminal[InventoryState], int], float] = {
        (s, a): 0. for s in si_mdp.non_terminal_states for a in si_mdp.actions(s)
    }
    
learning_rate_func: Callable[[int], float] = learning_rate_schedule(
    initial_learning_rate=initial_learning_rate,
    half_life=half_life,
    exponent=exponent
    )   

q_learning_result = q_learning(
    mdp = si_mdp,
    policy_from_q=lambda f, m: epsilon_greedy_policy(
            q=f,
            mdp=m,
            ϵ=q_learning_epsilon
        ),
    states=Choose(si_mdp.non_terminal_states),
    approx_0=Tabular(
        values_map=initial_qvf_dict,
        count_to_weight_func=learning_rate_func
    ),
    γ=gamma,
    max_episode_length=td_episode_length
)



print("MDP Policy Iteration Optimal Value Function and Optimal Policy for SimpleInventoryMDPCap")
print("--------------")
opt_vf_pi, opt_policy_pi = policy_iteration_result(
    si_mdp,
    gamma
)
pprint(opt_vf_pi)
print(opt_policy_pi)
print()

print("MDP Value Iteration Optimal Value Function and Optimal Policy for SimpleInventoryMDPCap")
print("--------------")
opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma)
pprint(opt_vf_vi)
print(opt_policy_vi)
print()
