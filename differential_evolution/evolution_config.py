from .evo_steps import *
from functools import partial

class DiffEvoConfig:
    def __init__(self):
        # population initialization
        self.init_population_size: int = 100
        self.init_strategy: str = "latin_hypercube"  # "uniform", "normal", or "latin_hypercube"
        self.init_bounds = (-1.0, 1.0)  # Search space bounds for initialization

        # common DE steps
        self.selection_strategy: str = "best"
        self.crossover_strategy: str = "exp"
        self.crossover_count: int = 2
        self.crossover_rate: float = 0.5

        self.differential_weight: float = 0.5     #F parameter
        self.max_generations: int    = 100
        self.tolerance: float        = 1e-6      # Convergence treshold

        self.noisy_best_noise_range    = (-1.0, +1.0)
        self.random_replacement_max_distance = 1.0

        # replacement mechanism
        self.replacement_strategy: str = ""
        self.replaced_count: int       = 20

        self.min_generations_before_convergence: int = 50
        self.patience: int = 10
    def __repr__(self):
        return (
            f"DE_sel={self.selection_strategy}"
            f"_xr={self.crossover_strategy}"
            f"_ncr={self.crossover_count}"
            f"_cr={self.crossover_rate:.2f}"
            f"_F={self.differential_weight:.2f}"
            f"_gens={self.max_generations}"
            f"_tol={self.tolerance}"
            f"_rep={self.replacement_strategy}{self.replaced_count}"
            f"_POP{self.init_population_size}_{self.init_strategy}"
        )
    
    def short_repr(self):
        return f"DE_{self.selection_strategy}_{self.crossover_strategy}_{self.crossover_count}_{self.replacement_strategy}_{self.replaced_count}_{self.init_strategy}_{self.init_population_size}"
    
    def param_optim_repr(self):
        """Represents parameters relevant in parameter optimization"""
        return f"Vanilla_pop{self.init_population_size}_CR{self.crossover_rate}_F{self.differential_weight}_tol{self.tolerance}"

    def get_init_strategy(self):
        return self.init_strategy

    def get_init_bounds(self):
        return self.init_bounds

    def get_init_population_size(self):
        return self.init_population_size

    def get_crossover_count(self):
        return self.crossover_count

    def get_crossover_rate(self):
        return self.crossover_rate

    def get_replaced_count(self):
        return self.replaced_count

    def get_differential_weight(self) -> float:
        return self.differential_weight

    def get_max_generations(self) -> int:
        return self.max_generations

    def get_tolerance(self) -> float:
        return self.tolerance

    def get_noisy_best_noise_range(self):
        return self.noisy_best_noise_range

    def get_random_replacement_max_distance(self):
        return self.random_replacement_max_distance

    def get_selection_fn(self):
        return selections_mapping[self.selection_strategy]

    def get_crossover_fn(self):
        return crossovers_mapping[self.crossover_strategy]

    def get_replacement_fn(self):
        base = replacements_mapping[self.replacement_strategy]
        if base is None:
            return None
        if self.replacement_strategy == "noisy_best":
            return partial(base, noise_range=self.get_noisy_best_noise_range())
        if self.replacement_strategy == "random":
            return partial(base, max_distance_per_idx=self.get_random_replacement_max_distance())
        return base

    def get_tournament_fn(self):
        return lambda x, y, obj_fn: min(x, y, key=obj_fn)
    

    def get_min_generations_before_convergence(self) -> int:
        return self.min_generations_before_convergence

    def get_patience(self) -> int:
        return self.patience

selections_mapping = {
    "best":   best_selection,
    "random": random_selection,
}

crossovers_mapping = {
    "bin": binary_crossover,
    "exp": exponential_crossover,
}

replacements_mapping = {
    "":           None,
    "noisy_best": noisy_best_replacement,
    "random":     random_replacement,
}
