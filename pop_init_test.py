from benchmark import run_suite, DiffEvoConfig
import cocopp

import os

def create_vanilla_cfg():
    cfg = DiffEvoConfig()
    cfg.replaced_count = 0
    cfg.replacement_strategy = ""
    
    cfg.min_generations_before_convergence = 5
    cfg.tolerance = 1e-8

    cfg.init_strategy = "normal"
    cfg.init_bounds = (-10.0, 10.0)
    cfg.init_population_size = 100
    
    cfg.differential_weight = 0.5
    cfg.crossover_rate = 0.9

    return cfg

def init_strategy_tests(seed):
    vanilla_cfg = create_vanilla_cfg()
    folders = []

    for init_strategy in ["uniform", "normal", "latin_hypercube"]:
        vanilla_cfg.init_strategy = init_strategy

        output_folder = f"V_ini_{init_strategy[:2]}{vanilla_cfg.init_bounds[1]}"
        run_suite(vanilla_cfg, output_folder, output_folder, seed=seed, postprocess=False)
        folders.append(output_folder)

    return folders


def main():
    SEED = 121
    strategy_folders = init_strategy_tests(SEED)

    # change dir to avoid ugly output filename
    os.chdir("exdata")
    # post-process each variable test
    for folder in [strategy_folders]:
        cocopp.main(" ".join([f"{exp}" for exp in folder]))

if __name__ == "__main__":
     main()