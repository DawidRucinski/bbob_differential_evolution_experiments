from .evo_steps import *


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

        # replacement mechanism
        self.replacement_strategy: str = ""
        self.replaced_count = 20

    def __repr__(self):
        return f"DE_{self.selection_strategy}_{self.crossover_strategy}_{self.crossover_count}_{self.replacement_strategy}_{self.replaced_count}_POP{self.init_population_size}_{self.init_strategy}"
    
    # Add getter methods for the new parameters
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
    
    def get_selection_fn(self):
        return selections_mapping[self.selection_strategy]

    def get_crossover_fn(self):
        return crossovers_mapping[self.crossover_strategy]

    def get_replacement_fn(self):
        return replacements_mapping[self.replacement_strategy]

    def get_tournament_fn(self):
        return lambda x, y, obj_fn: min(x, y, key=obj_fn)


selections_mapping = {
    "best": best_selection,
    "random": random_selection,
}

crossovers_mapping = {
    "bin": binary_crossover,
    "exp": exponential_crossover,
}

replacements_mapping = {
    "": None,
    "noisy_best": noisy_best_replacement,
    "random" : random_replacement,
}
