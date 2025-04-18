from differential_evolution import DiffEvoConfig, DiffEvoMinimizer


def main():
    config = DiffEvoConfig()

    config.init_population_size = 100

    opt = DiffEvoMinimizer(config=config)

    def test_fn(x): return sum([(xi-2)**2 for xi in x])
    result = opt(test_fn, dimensionality=3)

    print(result)

if __name__ == "__main__":
    main()
