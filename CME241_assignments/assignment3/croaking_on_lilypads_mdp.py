from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess, NonTerminal
from rl.distribution import Categorical
import itertools
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class LilypadState:
    current_lilypad: int


CroakLilypadMapping = Mapping[
    LilypadState,
    Mapping[str, Categorical[Tuple[LilypadState, float]]]
]

class CroakOnLilypadMDP(FiniteMarkovDecisionProcess[LilypadState, str]):

    def __init__(
        self,
        lilypad_number: int,
    ):
        self.lilypad_number: int = lilypad_number
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> CroakLilypadMapping:
        d: Dict[LilypadState, Dict[str, Categorical[Tuple[LilypadState,
                                                            float]]]] = {}
        
        for i in range(self.lilypad_number - 1):
            d1: Dict[str, Categorical[Tuple[LilypadState, float]]] = {}
            sr_probs_dict: Dict[LilypadState, float] = {}
            target_i = i + 1
            state: LilypadState = LilypadState(target_i)
            if target_i - 1 == self.lilypad_number:
                sr_probs_dict[(LilypadState(target_i-1),1.0)] = target_i /self.lilypad_number
            else:
                sr_probs_dict[(LilypadState(target_i-1),0.0)] = target_i /self.lilypad_number
            if target_i + 1 == self.lilypad_number:
                sr_probs_dict[(LilypadState(target_i+1), 1.0)] = (self.lilypad_number - target_i)/self.lilypad_number 
            else:
                sr_probs_dict[(LilypadState(target_i+1), 0.0)] = (self.lilypad_number - target_i)/self.lilypad_number 
            d1["A"] = Categorical(sr_probs_dict)
            sr_probs_dict = {}
            for j in range(self.lilypad_number + 1):
                if j != target_i:
                    if j == self.lilypad_number:
                        sr_probs_dict[(LilypadState(j), 1.0)] = 1/self.lilypad_number 
                    else:
                        sr_probs_dict[(LilypadState(j), 0.0)] = 1/self.lilypad_number 
            d1["B"] = Categorical(sr_probs_dict)
            d[state] = d1
        return d


if __name__ == '__main__':
    from pprint import pprint

    user_lilypad_number = 9
    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[LilypadState, str] =\
        CroakOnLilypadMDP(
            lilypad_number=user_lilypad_number,
        )

    print("MDP Transition Map")
    print("------------------")
    print(si_mdp)
    
    
    action_combination =list(itertools.product("AB", repeat = user_lilypad_number - 1))
    print(action_combination)
    

    for i in range(2 ** (user_lilypad_number - 1)):
        Policy = {}
        for j in range(user_lilypad_number - 1):
            Policy[LilypadState(j+1)] = action_combination[i][j]
        print("Deterministic Policy")
        print("--------------------")
        print(Policy)
        print()
        
        fdp: FiniteDeterministicPolicy[LilypadState, str] = \
            FiniteDeterministicPolicy(Policy)

        print("Deterministic Policy Map")
        print("------------------------")
        print(fdp)

        implied_mrp: FiniteMarkovRewardProcess[LilypadState] =\
            si_mdp.apply_finite_policy(fdp)
    
        print("Implied MRP Value Function")
        print("--------------")
        implied_mrp.display_value_function(gamma=user_gamma)
        print()

    from rl.dynamic_programming import evaluate_mrp_result
    from rl.dynamic_programming import policy_iteration_result
    from rl.dynamic_programming import value_iteration_result

    print("Implied MRP Policy Evaluation Value Function")
    print("--------------")
    pprint(evaluate_mrp_result(implied_mrp, gamma=user_gamma))
    print()

    print("MDP Policy Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_pi, opt_policy_pi = policy_iteration_result(
        si_mdp,
        gamma=user_gamma
    )
    pprint(opt_vf_pi)
    print(opt_policy_pi)
    print()

    print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma=user_gamma)
    pprint(opt_vf_vi)
    print(opt_policy_vi)
    print()
    
    
    x_data = []
    y_data = []
    
    for i in range(user_lilypad_number-1): 
        x_data.append(i+1)
        y_data.append(opt_vf_vi[NonTerminal(state=LilypadState(current_lilypad=i+1))])
    
    
    print(x_data)
    print(y_data)

    x = np.array(x_data)
    y = np.array(y_data)

    plt.plot(x, y, linestyle='-')

    # Add Title

    plt.title("Optimal Value Function")
    
    # Set Axes Range
    
    plt.xlim(1, user_lilypad_number - 1)
    plt.ylim(min(y_data) - 0.1, max(y_data) + 0.1)
    
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))

    # Add Axes Labels

    plt.xlabel("Lilypad Number")
    plt.ylabel("Optimal Escape-Probability")

    # Display

    plt.show()

    x_data = []
    y_data = []
    
    for x in range(-20,20):
        if x> 0:
            x_data.append(x)
            y_data.append( 0.25*(1.05*x-1)/(-0.04*x-0.25*0.25*x))
    
    
    print(x_data)
    print(y_data)

    x = np.array(x_data)
    y = np.array(y_data)
    
    plt.plot(x, y, linestyle='-')

     
    plt.xlabel("Risk Aversion")
    plt.ylabel("z")

    # Display

    plt.show()
    
    
    
    
    