from differential_evolution import DiffEvoConfig, DiffEvoMinimizer


def main():
    config = DiffEvoConfig()

    config.init_population_size = 100

    opt = DiffEvoMinimizer(config=config)

    def test_fn(x): return sum([(xi-2)**2 for xi in x])
    end_population = opt(test_fn, dimensionality=3)

    for x in end_population:
        print(x)

if __name__ == "__main__":
    main()
