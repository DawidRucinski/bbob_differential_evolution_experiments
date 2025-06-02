import csv
import os
from datetime import datetime
from random import choices
import numpy as np
import time
from .evolution_config import DiffEvoConfig
from .population import SortedPopulation


class DiffEvoMinimizer:
    def __init__(self, config=None):
        self.config = config or DiffEvoConfig()
        self.init_population_size = self.config.get_init_population_size()
        self.crossover_count = self.config.get_crossover_count()
        self.crossover_rate = self.config.get_crossover_rate()
        self.differential_weight = self.config.get_differential_weight()
        self.max_generations = self.config.get_max_generations()
        self.tolerance = self.config.get_tolerance()
        self.replaced_count = self.config.get_replaced_count()
        self.min_gens = self.config.get_min_generations_before_convergence()
        self.patience = self.config.get_patience()

        self.selection = self.config.get_selection_fn()
        self.crossover = self.config.get_crossover_fn()
        self.replacement = self.config.get_replacement_fn()
        self.tournament = self.config.get_tournament_fn()

    def __call__(self, problem, dimensionality):
        # odczyt nazwy funkcji z obiektu COCO Problem
        func_name = problem.name
        self.config.set_function_name(func_name)

        pop = SortedPopulation(
            self.init_population_size,
            dimensionality,
            problem,  # używamy problem(x) → ocena
            init_strategy=self.config.get_init_strategy(),
            bounds=self.config.get_init_bounds()
        )

        best_score = pop.get_scores()[0]
        stall_counter = 0

        run_name = (
            f"{self.config.get_function_name()}_"
            f"{self.config.replacement_strategy or 'vanilla'}_"
            f"dim{dimensionality}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self._init_metrics_loggers(run_name)
        log_folder = os.path.join("logs_comparison", run_name)
        os.makedirs(log_folder, exist_ok=True)

        start = time.time()
        for gen in range(self.max_generations):
            for i, current in enumerate(pop.get_population()):
                r = self.selection(pop, problem)

                M = np.array(r, copy=True)
                for _ in range(self.crossover_count):
                    d1, e1 = choices(pop.get_population(), k=2)
                    M += self.differential_weight * (d1 - e1)

                trial = self.crossover(r, M, self.crossover_rate) if self.crossover_rate < 1.0 else M
                winner = self.tournament(current, trial, problem)
                pop.overwrite(i, winner)

            if self.replacement:
                pop = self.replacement(pop, problem, self.replaced_count)

            scores = pop.get_scores()
            new_best = scores[0]
            delta = abs(best_score - new_best)

            self._log_metrics(gen, pop)

            if gen + 1 >= self.min_gens and delta < self.tolerance:
                stall_counter += 1
            else:
                stall_counter = 0

            if stall_counter >= self.patience:
                break

            best_score = new_best

        elapsed = time.time() - start

        with open(os.path.join(log_folder, "summary.txt"), "w") as f:
            f.write(f"Function: {self.config.get_function_name()}\n")
            f.write(f"Execution time: {elapsed:.3f} seconds\n")
            f.write(f"Final best score: {best_score}\n")
            f.write(f"Final mean diversity: {self._last_mean_distance:.4f}\n")
            f.write(f"Final variance: {np.var(pop.get_scores()):.4e}\n")
            f.write(f"Final fitness spread: {np.mean(np.array(pop.get_scores()) - best_score):.4e}\n")

        self._finalize_metrics_loggers()
        return pop.get_n_best(1)[0]


    def _init_metrics_loggers(self, run_name):
        folder = os.path.join("logs_comparison", run_name)
        os.makedirs(folder, exist_ok=True)

        self._convergence_log = open(os.path.join(folder, "convergence.csv"), "w", newline="")
        self._diversity_log = open(os.path.join(folder, "diversity.csv"), "w", newline="")

        self._convergence_writer = csv.writer(self._convergence_log)
        self._diversity_writer = csv.writer(self._diversity_log)

        self._convergence_writer.writerow(["generation", "best_score"])
        self._diversity_writer.writerow(["generation", "mean_distance", "fitness_variance", "fitness_spread"])

    def _log_metrics(self, generation, population):
        scores = population.get_scores()
        best_score = scores[0]

        pop_array = np.array(population.get_population())
        centroid = np.mean(pop_array, axis=0)
        distances = np.linalg.norm(pop_array - centroid, axis=1)

        mean_distance = np.mean(distances)
        variance = np.var(scores)
        spread = np.mean(np.array(scores) - best_score)

        self._last_mean_distance = mean_distance

        self._convergence_writer.writerow([generation, best_score])
        self._diversity_writer.writerow([generation, mean_distance, variance, spread])

    def _finalize_metrics_loggers(self):
        self._convergence_log.close()
        self._diversity_log.close()
