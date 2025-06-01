from benchmark import run_suite, DiffEvoConfig
import cocopp
import os

def base_cfg():
    cfg = DiffEvoConfig()
    cfg.differential_weight = 0.4
    cfg.crossover_rate = 0.9
    cfg.init_population_size = 200
    cfg.init_strategy = "normal"
    cfg.min_generations_before_convergence = 5
    cfg.tolerance = 1e-8
    return cfg

def create_vanilla_cfg():
    cfg = base_cfg()
    cfg.replaced_count = 0
    cfg.replacement_strategy = ""
    return cfg

def create_cfg_with_replacement(strategy, replaced_fraction=0.2, hybrid_p=None):
    cfg = base_cfg()
    cfg.replacement_strategy = strategy
    cfg.replaced_count = int(replaced_fraction * cfg.init_population_size)
    if strategy == "hybrid" and hybrid_p is not None:
        cfg.hybrid_random_prob = hybrid_p
    return cfg

def main():
    SEED = 123
    output_folders = []

    # Vanilla DE
    vanilla_cfg = create_vanilla_cfg()
    run_suite(vanilla_cfg, "vanilla", "vanilla", seed=SEED, postprocess=False)
    output_folders.append("vanilla")

    # Random replacement
    random_cfg = create_cfg_with_replacement("random")
    run_suite(random_cfg, "random", "random", seed=SEED, postprocess=False)
    output_folders.append("random")

    # Noisy best
    noisy_cfg = create_cfg_with_replacement("noisy_best")
    run_suite(noisy_cfg, "noisy_best", "noisy_best", seed=SEED, postprocess=False)
    output_folders.append("noisy_best")

    # Hybrid strategy
    hybrid_cfg = create_cfg_with_replacement("hybrid", hybrid_p=0.5)
    run_suite(hybrid_cfg, "hybrid", "hybrid", seed=SEED, postprocess=False)
    output_folders.append("hybrid")

    # Evaluate using COCO
    os.chdir("exdata")
    cocopp.main(" ".join(output_folders))

if __name__ == "__main__":
    main()
