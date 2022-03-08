from typing import Sequence, Tuple, Mapping
import numpy as np
from operator import itemgetter
from itertools import groupby
from numpy.random import randint

S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]

def get_state_return_samples(
    data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]

def get_mc_value_function(
    state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    """
    Implement tabular MC Value Function compatible with the interface defined above.
    """
    sorted_samples = sorted(state_return_samples, key=itemgetter(0))
    v: ValueFunc = {}
    for i, j in groupby(sorted_samples, itemgetter(0)):
        value = []
        for _, k in j:
            value.append(k)
        v[i] = np.mean(value)
    return v

def get_state_reward_next_state_samples(
    data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i+1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    """
    Implement code that produces the probability transitions and the
    reward function compatible with the interface defined above.
    """
    sorted_samples = sorted(srs_samples, key=itemgetter(0))
    d = {}
    for i, j in groupby(sorted_samples, itemgetter(0)):
        l = []
        for _, r, s in j:
            l.append((r,s))
        d[i] = l
        
    prob_func: ProbFunc = {}
    reward_func: RewardFunc = {}
    
    for i , j in d.items():
        p = {}
        for s, l in groupby(sorted(j, key=itemgetter(1)), itemgetter(1)):
            if s != 'T':
                p[s] = len(list(l)) / len(j)
        prob_func[i] = p
        
    for i, j in d.items():
        reward = []
        for r, _ in j:
            reward.append(r)
        reward_func[i] = np.mean(reward)
    
    return (prob_func,reward_func)
        
    
def get_mrp_value_function(
    prob_func: ProbFunc,
    reward_func: RewardFunc
) -> ValueFunc:
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """
    v: ValueFunc = {}
    states = list(reward_func.keys())
    reward = []
    for s in states:
        reward.append(reward_func[s])
    probability = [[prob_func[s][s1] if s1 in prob_func[s] else 0.
                            for s1 in states] for s in states]
    vector = np.linalg.inv(np.eye(len(states))-np.array(probability)).dot(np.array(reward))  
    for i in range(len(states)):
        v[states[i]] = vector[i]
    return v
        

def get_td_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    num_updates: int = 300000,
    learning_rate: float = 0.3,
    learning_rate_decay: int = 30
) -> ValueFunc:
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """
    value : ValueFunc = {}
    ret = {}
    for i, _, _ in srs_samples:
        for j in set(i):
            ret[j] = [0.0]
    for i in range(num_updates):
        s, r, s1 = srs_samples[randint(len(srs_samples), size =1)[0]]
        if s1 == 'T':
            ret[s].append(ret[s][-1] + learning_rate * (i / learning_rate_decay + 1) ** -0.5 * (r + 0 - ret[s][-1]))
        else:
            ret[s].append(ret[s][-1] + learning_rate * (i / learning_rate_decay + 1) ** -0.5 * (r + ret[s1][-1] - ret[s][-1]))
    for s, v in ret.items():
        value[s] = np.mean(v[-int(len(v) * 0.9):])
        
    return value

def get_lstd_value_function(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """
    value : ValueFunc = {}
    states = []
    for i, _, _ in srs_samples:
        states.append(i)
    states = list(set(states))
    phi = np.eye(len(states))
    a = np.zeros((len(states), len(states)))
    b = np.zeros(len(states))
    for s, r, s1 in srs_samples:
        p1 = phi[states.index(s)]
        if s1 == "T":
            p2 = np.zeros(len(states))
        else:
            p2 = phi[states.index(s1)] 
        a += np.outer(p1, p1 - p2)
        b += p1 * r
    for s, v in enumerate(np.linalg.inv(a).dot(b)):
        value[states[s]] = v
    return value
        


if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    sr_samps = get_state_return_samples(given_data)

    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    srs_samps = get_state_reward_next_state_samples(given_data)

    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)
    print("-------------- MRP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))
    
    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))
    
    
    
    
    
    




