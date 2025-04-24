from random import choices
import numpy as np

from .evolution_config import DiffEvoConfig
from .population import SortedPopulation

class DiffEvoMinimizer:
    def __init__(self, config=None):
        self.config = config or DiffEvoConfig()

        self.init_population_size = config.get_init_population_size()
        self.crossover_count = config.get_crossover_count()
        self.crossover_rate = config.get_crossover_rate()
        self.replaced_count = config.get_replaced_count()

        self.selection:   function = config.get_selection_fn()
        self.crossover:   function = config.get_crossover_fn()
        self.replacement: function = config.get_replacement_fn()
        self.tournament:  function = config.get_tournament_fn()

    def __call__(self, objective_function, dimensionality):
        population = SortedPopulation(
            self.init_population_size, 
            dimensionality, 
            objective_function,
            init_strategy=self.config.get_init_strategy(),
            bounds=self.config.get_init_bounds()
        )

        for _ in range(25):  # TODO set actual stop condition
            for i, p in enumerate(population.get_population()):
                # working point
                r = self.selection(population, objective_function)

                # modifying r based on differences between pairs
                M = np.array(r)
                for _ in range(self.crossover_count):
                    d1, e1 = choices(population.get_population(), k=2)
                    F = 1/2  # TODO: parametrize
                    M += F*(d1 - e1)

                O = self.crossover(r, M, self.crossover_rate)
                population.overwrite(i, self.tournament(p, O, objective_function))

            if self.replacement:
                population = self.replacement(population, objective_function, self.replaced_count)

        return population.get_n_best(1)[0]
    