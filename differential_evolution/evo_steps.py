from random import random, choice
import numpy as np
from numpy import random as nprd

def best_selection(population, obj_fn):
    return min(population, key=obj_fn)

def random_selection(population, _):
    return choice(population)

def binary_crossover(x, y, cr):
    return np.array([yi if (random() < cr) else xi for (xi, yi) in zip(x, y)])

def exponential_crossover(x, y, cr):
    # calculate cutoff point (TODO: can be optimized by using exp distribution random fn)
    cutoff = 0
    while random() < cr and cutoff < len(x):
        cutoff += 1
    return np.concatenate((y[:cutoff], x[cutoff:]))

def noisy_best_replacement(population, obj_fn, replaced_count):
    # TODO: parametrize
    # how many of the best fitting entities should be used for replacement 
    num_best = replaced_count
    population = sorted(population, key=obj_fn)

 
    for i in range(replaced_count):
        copied_best = choice(population[:num_best])
        # TODO: parametrize
        # currently applies linearly-distributed noise from range [-1.0; +1.0]
        population[-i-1] = copied_best + 2.0*nprd.random(len(copied_best)) - 1.0

    return population

def random_replacement(population, obj_fn, replaced_count):
    population = np.array(sorted(population, key=obj_fn))

    # TODO: discuss and parametrize
    # maximum distance from furthest coordinate along an axis
    max_distance_per_idx = 1.0

    minimums = population.min(axis=0) - max_distance_per_idx
    maximums = population.max(axis=0) + max_distance_per_idx

    # random point from a uniform distribution on each attribute separately
    for i in range(replaced_count):
        population[-i-1] = minimums + nprd.random(size=minimums.size) * (maximums - minimums)

    return population