from benchmark import run_suite
from differential_evolution import DiffEvoConfig

def main():
    cfg = DiffEvoConfig()
    cfg.replaced_count = 0
    cfg.replacement_strategy = ""
    
    cfg.min_generations_before_convergence = 5
    cfg.tolerance = 1e-8

    cfg.init_strategy = "normal"
    cfg.init_bounds = (-10.0, 10.0)
    cfg.init_population_size = 100
    
    cfg.differential_weight = 0.4
    cfg.crossover_rate = 0.9
    




if __name__ == "__main__":
    main()
