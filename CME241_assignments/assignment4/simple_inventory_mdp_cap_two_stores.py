from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical
from scipy.stats import poisson


@dataclass(frozen=True)
class InventoryState:
    on_hand_a: int
    on_order_a: int
    on_hand_b: int
    on_order_b: int

    def inventory_position_a(self) -> int:
        return self.on_hand_a + self.on_order_a
        
    def inventory_position_b(self) -> int:
        return self.on_hand_b + self.on_order_b
    


InvOrderMapping = Mapping[
    InventoryState,
    Mapping[Tuple[int,int,int], Categorical[Tuple[InventoryState, float]]]
]


class SimpleInventoryTwoStoresMDPCap(FiniteMarkovDecisionProcess[InventoryState, int]):

    def __init__(
        self,
        capacity_a: int,
        capacity_b: int,
        poisson_lambda_a: float,
        poisson_lambda_b:float,
        holding_cost_a: float,
        holding_cost_b: float,
        stockout_cost_a: float,
        stockout_cost_b: float,
        trans_cost_supplier: float,
        trans_cost_stores: float
        
    ):
        self.capacity_a: int = capacity_a
        self.poisson_lambda_a: float = poisson_lambda_a
        self.holding_cost_a: float = holding_cost_a
        self.stockout_cost_a: float = stockout_cost_a
        self.capacity_b: int = capacity_b
        self.poisson_lambda_b: float = poisson_lambda_b
        self.holding_cost_b: float = holding_cost_b
        self.stockout_cost_b: float = stockout_cost_b
        self.trans_cost_supplier: float = trans_cost_supplier
        self.trans_cost_stores: float = trans_cost_stores

        self.poisson_distr_a = poisson(poisson_lambda_a)
        self.poisson_distr_b = poisson(poisson_lambda_b)
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[Tuple(int,int,int), Categorical[Tuple[InventoryState,
                                                            float]]]] = {}
        for alpha_a in range(self.capacity_a + 1):
            for beta_a in range(self.capacity_a + 1 - alpha_a):
                for alpha_b in range(self.capacity_b + 1):
                    for beta_b in range(self.capacity_b + 1 - alpha_b):
                        state: InventoryState = InventoryState(alpha_a, beta_a, alpha_b, beta_b)
                        ip_a: int = state.inventory_position_a()
                        ip_b: int = state.inventory_position_b()
                        d1: Dict[Tuple[int,int,int], Categorical[Tuple[InventoryState, float]]] = {}
                      
                        for order_e in range(max(-alpha_a, -(self.capacity_a-ip_a)), min(alpha_b, self.capacity_b-ip_b)+1):
                            for order_a in range(self.capacity_a-ip_a-order_e+1):
                                for order_b in range(self.capacity_b-ip_b+order_e+1):
                                    sr_probs_dict: Dict[Tuple[InventoryState, float], float] = {}
                                    base_reward: float = - self.holding_cost_a * alpha_a -self.holding_cost_b * alpha_b
                                    if order_a > 0:
                                        base_reward -= self.trans_cost_supplier
                                    if order_b > 0:
                                        base_reward -= self.trans_cost_supplier
                                    if order_e != 0:
                                        base_reward -= self.trans_cost_stores
                                    if order_e > 0:
                                        base_reward += self.holding_cost_b * order_e
                                    if order_e < 0 :
                                        base_reward -= self.holding_cost_a * order_e
                                    for demand_a in range(ip_a + order_e):
                                        for demand_b in range(ip_b  - order_e):
                                            sr_probs_dict[(InventoryState(ip_a + order_e - demand_a,order_a,ip_b - order_e - demand_b, order_b), base_reward)] = self.poisson_distr_a.pmf(demand_a)* self.poisson_distr_b.pmf(demand_b)        
                                    for demand_a in range(ip_a + order_e):
                                        probability: float = 1 - self.poisson_distr_b.cdf(ip_b - order_e - 1)
                                        reward: float = base_reward - self.stockout_cost_b *\
                                            (probability * (self.poisson_lambda_b - ip_b + order_e) +
                                                 (ip_b - order_e) * self.poisson_distr_b.pmf(ip_b - order_e))
                                        sr_probs_dict[(InventoryState(ip_a + order_e - demand_a,order_a, 0 , order_b), reward)] = self.poisson_distr_a.pmf(demand_a)* (1-self.poisson_distr_b.cdf(ip_b - order_e - 1))
                                    for demand_b in range(ip_b - order_e):
                                        probability: float = 1 - self.poisson_distr_a.cdf(ip_a + order_e - 1)
                                        reward: float = base_reward - self.stockout_cost_a *\
                                            (probability * (self.poisson_lambda_a - ip_a - order_e) +
                                                 (ip_a + order_e) * self.poisson_distr_a.pmf(ip_a + order_e))
                                        sr_probs_dict[(InventoryState(0 ,order_a, ip_b - order_e - demand_b, order_b), reward)] = self.poisson_distr_b.pmf(demand_b)* (1-self.poisson_distr_a.cdf(ip_a + order_e - 1))
                                    reward = base_reward - self.stockout_cost_a *\
                                            ( (1-self.poisson_distr_a.cdf(ip_a + order_e - 1))* (self.poisson_lambda_a - ip_a - order_e) + (ip_a + order_e) * self.poisson_distr_a.pmf(ip_a + order_e))- \
                                            self.stockout_cost_b * ((1-self.poisson_distr_b.cdf(ip_b - order_e - 1)) * (self.poisson_lambda_b - ip_b + order_e) +(ip_b - order_e) * self.poisson_distr_b.pmf(ip_b - order_e))
                                    sr_probs_dict[(InventoryState(0 ,order_a, 0 , order_b), reward)] = (1-self.poisson_distr_b.cdf(ip_b - order_e - 1)) * (1-self.poisson_distr_a.cdf(ip_a + order_e - 1))
                                    d1[(order_a, order_b, order_e)] = Categorical(sr_probs_dict)
                        d[state] = d1
        return d


if __name__ == '__main__':
    from pprint import pprint

    user_capacity_a = 5
    user_capacity_b = 5
    user_poisson_lambda_a = 1.0
    user_poisson_lambda_b = 1.0
    user_holding_cost_a = 1.0
    user_holding_cost_b = 1.0
    user_stockout_cost_a = 10.0
    user_stockout_cost_b = 10.0
    user_trans_cost_supplier = 5.0
    user_trans_cost_stores = 7.0
    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
        SimpleInventoryTwoStoresMDPCap(
            capacity_a = user_capacity_a,
            capacity_b = user_capacity_b,
            poisson_lambda_a = user_poisson_lambda_a,
            poisson_lambda_b= user_poisson_lambda_b,
            holding_cost_a = user_holding_cost_a,
            holding_cost_b = user_holding_cost_b,
            stockout_cost_a = user_stockout_cost_a,
            stockout_cost_b = user_stockout_cost_b,
            trans_cost_supplier = user_trans_cost_supplier,
            trans_cost_stores = user_trans_cost_stores 
        )

    print("MDP Transition Map")
    print("------------------")
    print(si_mdp)
    

    from rl.dynamic_programming import policy_iteration_result
    from rl.dynamic_programming import value_iteration_result



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
