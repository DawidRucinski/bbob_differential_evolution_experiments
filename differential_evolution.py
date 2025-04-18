from random import random, choices
from evolution_config import DiffEvoConfig


def diff_multiplied(a: list, b: list, multiplier: float):
    return [multiplier*(x1 - x2) for (x1, x2) in zip(a, b)]


def arr_sum(a: list, b: list):
    return [x + y for x, y in zip(a, b)]


class DiffEvoMinimizer:
    def __init__(self, config=None):
        self.population_init: function = random

        self.config = config or DiffEvoConfig()
        self.selection:   function = config.get_selection_fn()
        self.crossover:   function = config.get_crossover_fn()
        self.replacement: function = config.get_replacement_fn()
        self.tournament:  function = config.get_tournament_fn()

    def __call__(self, objective_function, dimensionality):
        population = [[self.population_init() for _ in range(dimensionality)]
                      for i in range(self.config.init_population_size)]

        for _ in range(10):  # TODO set actual stop condition
            for i, p in enumerate(population):
                # working point
                r = self.selection(population, objective_function)

                # modifying r based on differences between pairs
                M = r
                for _ in range(self.config.crossover_count):
                    d1, e1 = choices(population, k=2)
                    F = 1/2  # TODO: parametrize
                    M = arr_sum(M, diff_multiplied(d1, e1, F))

                O = self.crossover(r, M, self.config.crossover_rate)
                population[i] = self.tournament(p, O, objective_function)

        return population
