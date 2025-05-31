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
        self.init_population_size      = self.config.get_init_population_size()
        self.crossover_count           = self.config.get_crossover_count()
        self.crossover_rate            = self.config.get_crossover_rate()
        self.differential_weight       = self.config.get_differential_weight()
        self.max_generations           = self.config.get_max_generations()
        self.tolerance                 = self.config.get_tolerance()
        self.replaced_count            = self.config.get_replaced_count()
        self.min_gens                  = self.config.get_min_generations_before_convergence()
        self.patience                  = self.config.get_patience()

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

        best_score    = pop.get_scores()[0]
        stall_counter = 0

        # prepare logging
        os.makedirs("logs", exist_ok=True)
        converg_path = os.path.join("logs", "convergence.csv")
        divers_path  = os.path.join("logs", "diversity.csv")
        summary_path = os.path.join("logs", "summary.txt")

        with open(converg_path, "w", newline="") as f_conv, open(divers_path, "w", newline="") as f_div:
            conv_writer = csv.writer(f_conv)
            div_writer = csv.writer(f_div)
            conv_writer.writerow(["generation", "best_score"])
            div_writer.writerow(["generation", "mean_distance", "fitness_variance", "fitness_spread"])

            start = time.time()
            for gen in range(self.max_generations):
                for i, current in enumerate(pop.get_population()):
                    r = self.selection(pop, objective_function)

                    M = np.array(r, copy=True)
                    for _ in range(self.crossover_count):
                        d1, e1 = choices(pop.get_population(), k=2)
                        M += self.differential_weight * (d1 - e1)

                    trial = self.crossover(r, M, self.crossover_rate) if self.crossover_rate < 1.0 else M
                    winner = self.tournament(current, trial, objective_function)
                    pop.overwrite(i, winner)

                if self.replacement:
                    pop = self.replacement(pop, objective_function, self.replaced_count)

                scores = pop.get_scores()
                new_best = scores[0]
                delta = abs(best_score - new_best)

                # diversity metrics
                positions = pop.get_population()
                centroid = np.mean(positions, axis=0)
                mean_distance = np.mean(np.linalg.norm(positions - centroid, axis=1))
                variance = np.var(scores)
                spread = np.mean(np.array(scores) - new_best)

                # write logs
                conv_writer.writerow([gen, new_best])
                div_writer.writerow([gen, mean_distance, variance, spread])

                if gen + 1 >= self.min_gens and delta < self.tolerance:
                    stall_counter += 1
                else:
                    stall_counter = 0

                if stall_counter >= self.patience:
                    print(
                        f"Stopped at generation {gen+1} "
                        f"(no improvement > {self.tolerance} for {self.patience} gens, Δ={delta:.3e})"
                    )
                    break

                best_score = new_best

            elapsed = time.time() - start
            with open(summary_path, "w") as f:
                f.write(f"Execution time: {elapsed:.3f} seconds\n")
                f.write(f"Final best score: {new_best}\n")
                f.write(f"Final mean diversity: {mean_distance:.4f}\n")
                f.write(f"Final variance: {variance:.4e}\n")
                f.write(f"Final fitness spread: {spread:.4e}\n")

        return pop.get_n_best(1)[0]


    def _init_metrics_loggers(self, run_name):
        folder = os.path.join("metrics_logs", run_name)
        os.makedirs(folder, exist_ok=True)

        convergence_path = os.path.join(folder, "convergence.csv")
        diversity_path   = os.path.join(folder, "diversity.csv")

        self._convergence_log = open(convergence_path, "w", newline="")
        self._diversity_log   = open(diversity_path, "w", newline="")

        self._convergence_writer = csv.writer(self._convergence_log)
        self._diversity_writer = csv.writer(self._diversity_log)

        self._convergence_writer.writerow(["generation", "best_score"])
        self._diversity_writer.writerow(["generation", "mean_distance"])

    def _log_metrics(self, generation, population):
        best_score = population.get_scores()[0]
        self._convergence_writer.writerow([generation, best_score])

        pop_array = np.array(population.get_population())
        centroid = np.mean(pop_array, axis=0)
        distances = np.linalg.norm(pop_array - centroid, axis=1)
        mean_distance = np.mean(distances)
        self._diversity_writer.writerow([generation, mean_distance])


    def _finalize_metrics_loggers(self):
        self._convergence_log.close()
        self._diversity_log.close()
    