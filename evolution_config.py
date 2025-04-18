from evo_steps import *


class DiffEvoConfig:
    def __init__(self):
        # population initialization
        self.init_population_size: int = 100
        self.population_init_fn: str = ""

        # common DE steps
        self.selection_strategy: str = "best"
        self.crossover_strategy: str = "exp"
        self.crossover_count: int = 2
        self.crossover_rate: float = 0.5

        # replacement mechanism
        self.replacement_strategy: str = ""

    def get_init_population_size(self):
        return self.init_population_size

    def get_crossover_count(self):
        return self.crossover_count

    def get_crossover_rate(self):
        return self.crossover_rate

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
    "": lambda x: x,
}
