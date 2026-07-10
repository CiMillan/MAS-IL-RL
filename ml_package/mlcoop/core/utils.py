import numpy as np


def make_rng(seed):
    return np.random.default_rng(seed)


def initialize_strategies(N, coop_fraction, rng):
    return (rng.random(N) < coop_fraction).astype(np.int8)


def fraction_cooperators(strategies):
    return float(np.mean(strategies))


def softmax_binary(v_c, v_d, beta):
    m = np.maximum(beta * v_c, beta * v_d)
    e_c = np.exp(beta * v_c - m)
    e_d = np.exp(beta * v_d - m)
    return e_c / (e_c + e_d)
