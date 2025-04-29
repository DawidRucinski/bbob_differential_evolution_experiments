# evolution_impl.py

from random import choices
import numpy as np

from .evolution_config import DiffEvoConfig
from .population import SortedPopulation

class DiffEvoMinimizer:
    def __init__(self, config=None):
        self.config = config or DiffEvoConfig()

        # pull all parameters from config
        self.init_population_size = self.config.get_init_population_size()
        self.crossover_count    = self.config.get_crossover_count()
        self.crossover_rate     = self.config.get_crossover_rate()
        self.differential_weight= self.config.get_differential_weight()
        self.max_generations    = self.config.get_max_generations()
        self.tolerance          = self.config.get_tolerance()
        self.replaced_count     = self.config.get_replaced_count()

        self.selection   = self.config.get_selection_fn()
        self.crossover   = self.config.get_crossover_fn()
        self.replacement = self.config.get_replacement_fn()
        self.tournament  = self.config.get_tournament_fn()

    def __call__(self, objective_function, dimensionality):
        pop = SortedPopulation(
            self.init_population_size,
            dimensionality,
            objective_function,
            init_strategy=self.config.get_init_strategy(),
            bounds=self.config.get_init_bounds()
        )

        best_score = pop.get_scores()[0]
        for gen in range(self.max_generations):
            for i, current in enumerate(pop.get_population()):
                r = self.selection(pop, objective_function)
                M = np.array(r, copy=True)
                for _ in range(self.crossover_count):
                    d1, e1 = choices(pop.get_population(), k=2)
                    M += self.differential_weight * (d1 - e1)
                trial = self.crossover(r, M, self.crossover_rate)
                winner = self.tournament(current, trial, objective_function)
                pop.overwrite(i, winner)

            if self.replacement:
                pop = self.replacement(pop, objective_function, self.replaced_count)
            #Early stopping - to discuss
            new_best = pop.get_scores()[0]
            if abs(best_score - new_best) < self.tolerance:
                print(f"Converged at generation {gen+1} (Δ = {abs(best_score-new_best):.3e})")
                break
            best_score = new_best

        return pop.get_n_best(1)[0]
