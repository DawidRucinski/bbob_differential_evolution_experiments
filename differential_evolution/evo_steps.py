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
    best_entities = population.get_n_best(num_best)
    low, high = noise_range
   
    def create_replacement():
        copied_best = choice(best_entities)
        noise = low + (high - low) * nprd.random(len(copied_best))
        
        return copied_best + noise
        
    replacers = [create_replacement() for _ in range(replaced_count)]
    population.replace_range(population.len() - replaced_count, replacers)


    return population


def random_replacement(population, obj_fn, replaced_count, max_distance_per_idx=1.0):
    minimums = population.get_population().min(axis=0) - max_distance_per_idx
    maximums = population.get_population().max(axis=0) + max_distance_per_idx
    
    def create_replacement():
        return minimums + nprd.random(size=minimums.size) * (maximums - minimums)
    
    replacers = [create_replacement() for _ in range(replaced_count)]
    population.replace_range(population.len() - replaced_count, replacers)
    
    return population

def hybrid_replacement(population, obj_fn, replaced_count, noise_range=(-1.0, 1.0), max_distance_per_idx=1.0, p_random=0.5):
    num_best = replaced_count
    best_entities = population.get_n_best(num_best)
    low, high = noise_range

    pop_array = population.get_population()
    pop_min = pop_array.min(axis=0) - max_distance_per_idx
    pop_max = pop_array.max(axis=0) + max_distance_per_idx

    def create_noisy_best():
        copied_best = choice(best_entities)
        noise = low + (high - low) * nprd.random(len(copied_best))
        return copied_best + noise

    def create_random():
        return pop_min + nprd.random(size=pop_min.size) * (pop_max - pop_min)

    replacers = [
        create_random() if nprd.random() < p_random else create_noisy_best()
        for _ in range(replaced_count)
    ]

    population.replace_range(population.len() - replaced_count, replacers)
    return population