from differential_evolution import DiffEvoConfig, DiffEvoOptimizer


def main():
    config = DiffEvoConfig()

    config.init_population_size = 100

    opt = DiffEvoOptimizer(config=config)

    def test_fn(x): return sum([-(xi-2)**2 for xi in x])
    end_population = opt(test_fn, dimensionality=3)

    print(end_population)

if __name__ == "__main__":
    main()
