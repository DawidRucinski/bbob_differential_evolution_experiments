from random import random, choice

def best_selection(population, obj_fn):
    return min(population, key=obj_fn)

def random_selection(population, _):
    return choice(population)

def binary_crossover(x, y, cr):
    return [yi if (random() < cr) else xi for (xi, yi) in zip(x, y)]

def exponential_crossover(x, y, cr):
    # calculate cutoff point (TODO: can be optimized by using exp distribution random fn)
    cutoff = 0
    while random() < cr and cutoff < len(x):
        cutoff += 1
    
    return y[:cutoff] + x[cutoff:]
