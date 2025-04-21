from differential_evolution import DiffEvoConfig, DiffEvoMinimizer
from numpy import random as nprd
import random as rd


def main():
    rd.seed(13)
    nprd.seed(12)  

    config = DiffEvoConfig()

    config.init_population_size = 100
    config.replacement_strategy = "noisy_best"

    print(config)

    opt = DiffEvoMinimizer(config=config)

    def test_fn(x): return sum([(xi-2)**2 for xi in x])
    result = opt(test_fn, dimensionality=3)

    print(result)

if __name__ == "__main__":
    main()
