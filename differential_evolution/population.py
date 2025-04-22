import numpy.random as rd
import numpy as np

class SortedPopulation:
    def __init__(self, count, dimensionality, objective_fn):
        self.objective_fn = objective_fn

        # TODO: decide on and parametrize population initialization
        self.population = 2 * rd.random(size=(count, dimensionality)) - 1.0
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

    def get_n_best(self, n):
        return self.population[:n]