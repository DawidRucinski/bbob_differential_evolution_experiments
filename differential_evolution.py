from random import random, choices
from evolution_config import DiffEvoConfig
import numpy as np
from numpy import random as rd

class DiffEvoMinimizer:
    def __init__(self, config=None):
        self.population_init: function = random

        self.config = config or DiffEvoConfig()
        self.selection:   function = config.get_selection_fn()
        self.crossover:   function = config.get_crossover_fn()
        self.replacement: function = config.get_replacement_fn()
        self.tournament:  function = config.get_tournament_fn()

    def __call__(self, objective_function, dimensionality):
        population = 2 * rd.random(size=(self.config.init_population_size, dimensionality)) - 1.0

        for _ in range(10):  # TODO set actual stop condition
            for i, p in enumerate(population):
                # working point
                r = self.selection(population, objective_function)

                # modifying r based on differences between pairs
                M = np.array(r)
                for _ in range(self.config.crossover_count):
                    d1, e1 = choices(population, k=2)
                    F = 1/2  # TODO: parametrize
                    M += F*(d1 - e1)

                O = self.crossover(r, M, self.config.crossover_rate)
                population[i] = self.tournament(p, O, objective_function)

        return min(population, key=objective_function)
