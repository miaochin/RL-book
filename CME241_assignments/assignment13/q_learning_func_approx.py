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
from rl.function_approx import LinearFunctionApprox, Weights, Tabular, learning_rate_schedule, DNNSpec
import rl.iterate as iterate
import rl.markov_process as mp
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_decision_process import TransitionStep, NonTerminal
from rl.monte_carlo import greedy_policy_from_qvf
from rl.policy import Policy, DeterministicPolicy
from rl.experience_replay import ExperienceReplayMemory
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap, InventoryState
from rl.distribution import Choose, Gaussian
from rl.dynamic_programming import evaluate_mrp_result
from rl.dynamic_programming import policy_iteration_result
from rl.dynamic_programming import value_iteration_result
from rl.chapter7.asset_alloc_discrete import AssetAllocDiscrete
from pprint import pprint

gamma: float = 0.9
q_learning_epsilon: float = 0.2

steps: int = 4
μ: float = 0.13
σ: float = 0.2
r: float = 0.07
a: float = 1.0
init_wealth: float = 1.0
init_wealth_stdev: float = 0.1
excess: float = μ - r
var: float = σ * σ
base_alloc: float = excess / (a * var)

risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
riskless_ret: Sequence[float] = [r for _ in range(steps)]
utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a

alloc_choices: Sequence[float] = np.linspace(
    2 / 3 * base_alloc,
    4 / 3 * base_alloc,
    11
)

feature_funcs: Sequence[Callable[[Tuple[float, float]], float]] = \
    [
        lambda _: 1.,
        lambda w_x: w_x[0],
        lambda w_x: w_x[1],
        lambda w_x: w_x[1] * w_x[1]
    ]
    
dnn: DNNSpec = DNNSpec(
    neurons=[],
    bias=False,
    hidden_activation=lambda x: x,
    hidden_activation_deriv=lambda y: np.ones_like(y),
    output_activation=lambda x: - np.sign(a) * np.exp(-x),
    output_activation_deriv=lambda y: -y
)

init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_stdev)

aad: AssetAllocDiscrete = AssetAllocDiscrete(
    risky_return_distributions=risky_ret,
    riskless_returns=riskless_ret,
    utility_func=utility_function,
    risky_alloc_choices=alloc_choices,
    feature_functions=feature_funcs,
    dnn_spec=dnn,
    initial_wealth_distribution=init_wealth_distr
)

epsilon_as_func_of_episodes: Callable[[float], float] = lambda k: k ** -0.5

q_learning_result = q_learning(
    mdp=aad.get_mdp(steps),
    policy_from_q=lambda f, m: epsilon_greedy_policy(
            q=f,
            mdp=m,
            ϵ=q_learning_epsilon
        ),
    states=init_wealth_distr.sample(),
    approx_0=aad.get_qvf_func_approx(),
    γ=gamma,
    max_episode_length=steps
) 




