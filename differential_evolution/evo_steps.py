from random import random, choice
import numpy as np
from numpy import random as nprd

def best_selection(population, obj_fn):
    return population.get_n_best(1)[0]

def random_selection(population, _):
    return choice(population.get_population())

def binary_crossover(x, y, cr):
    return np.array([yi if (random() < cr) else xi for (xi, yi) in zip(x, y)])

def exponential_crossover(x, y, cr):
    # calculate cutoff point - modified to use geometric sampling
    cutoff = min(len(x), nprd.geometric(p=1 - cr))  
    return np.concatenate((y[:cutoff], x[cutoff:]))

def noisy_best_replacement(population, obj_fn, replaced_count, noise_range=(-1.0, 1.0)):
    num_best = replaced_count
    replacers = population.get_n_best(num_best)
    low, high = noise_range
    for i in range(replaced_count):
        copied_best = choice(replacers)
        noise = low + (high - low) * nprd.random(len(copied_best))
        population.overwrite(-i-1, copied_best + noise)
    return population


def random_replacement(population, obj_fn, replaced_count, max_distance_per_idx=1.0):
    minimums = population.get_population().min(axis=0) - max_distance_per_idx
    maximums = population.get_population().max(axis=0) + max_distance_per_idx
    for i in range(replaced_count):
        random_point = minimums + nprd.random(size=minimums.size) * (maximums - minimums)
        population.overwrite(-i-1, random_point)
    return population