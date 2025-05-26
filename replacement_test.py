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
    cfg.init_population_size = 200
    
    cfg.differential_weight = 0.5
    cfg.crossover_rate = 0.9

    return cfg

def create_replacement_cfg(replacement_strategy, replaced_fraction):
    cfg = create_vanilla_cfg()

    cfg.replacement_strategy = replacement_strategy
    cfg.replaced_count = int(replaced_fraction*cfg.init_population_size)

    return cfg

def alg_name(replacement_strategy: str):
    if replacement_strategy == "random":
        return "rr"
    elif replacement_strategy == "noisy_best":
        return "nb"
    else:
        return "unknown_alg"

def replaced_fraction_tests(replacement_strategy, seed):
    cfg = create_vanilla_cfg()
    cfg.replacement_strategy = replacement_strategy

    folders = []

    for fract in [0.1, 0.2, 0.3, 0.4, 0.5]:
        cfg.replaced_count = int(fract*cfg.init_population_size)

        output_folder = f"{alg_name(replacement_strategy)}_fract{fract}"
        run_suite(cfg, output_folder, output_folder, seed=seed, postprocess=False)
        folders.append(output_folder)

    return folders

def replaced_distance_tests(replacement_strategy, seed):
    cfg = create_vanilla_cfg()
    cfg.replacement_strategy = replacement_strategy
    cfg.replaced_count = int(0.2*cfg.init_population_size)

    folders = []

    for distance in [5.0, 10.0, 20.0, 50.0, 100.0]:
        cfg.random_replacement_max_distance = distance
        cfg.noisy_best_noise_range = (-distance, distance)

        output_folder = f"{alg_name(replacement_strategy)}_d{distance}"
        run_suite(cfg, output_folder, output_folder, seed=seed, postprocess=False)
        folders.append(output_folder)

    return folders

def main():
    SEED = 121
    rr_fraction_folders = replaced_fraction_tests("random", SEED)
    nb_fraction_folders = replaced_fraction_tests("noisy_best", SEED)
    
    rr_distance_folders = replaced_distance_tests("random", SEED)
    nb_distance_folders = replaced_distance_tests("noisy_best", SEED)

    
    # change dir to avoid ugly output filename
    os.chdir("exdata")
    # post-process each variable test
    
    for folder in [rr_fraction_folders, nb_fraction_folders, rr_distance_folders, nb_distance_folders]:
        cocopp.main(" ".join([f"{exp}" for exp in folder]))

if __name__ == "__main__":
     main()