import numpy.random as rd
import numpy as np
from .population_init import init_strategies

class SortedPopulation:
    def __init__(self, count, dimensionality, objective_fn, init_strategy="uniform", bounds=(-1.0, 1.0)):
        self.objective_fn = objective_fn
        
        # Get the initialization function from the dictionary
        init_fn = init_strategies.get(init_strategy, init_strategies["uniform"])
        
        # Initialize population using the selected strategy
        self.population = init_fn(count, dimensionality, bounds)
        self.scores = np.array([objective_fn(x) for x in self.population])
        self.sort()

    def sort(self):
        sort_idxs = self.scores.argsort()
        self.population, self.scores = self.population[sort_idxs], self.scores[sort_idxs]

    def get_population(self):
        return self.population        
    
    def get_scores(self):
        return self.scores
    
    def overwrite(self, i, entity):
        self.population = np.delete(self.population, i, axis=0)
        self.scores = np.delete(self.scores, i, axis=0)

        entity_score = self.objective_fn(entity)

        insert_idx = self.scores.searchsorted(entity_score)

        self.population = np.insert(self.population, insert_idx, entity, axis=0)
        self.scores = np.insert(self.scores, insert_idx, entity_score, axis=0)


    def replace_range(self, start_idx, entities):
        assert (len(entities) + start_idx) <= len(self.population) 

        for i, entity in enumerate(entities):
            self.population[start_idx + i] = entity
            self.scores[start_idx + i] = self.objective_fn(entity)

        self.sort()

    def get_n_best(self, n):
        return self.population[:n]
    
    def len(self):
        return len(self.population)