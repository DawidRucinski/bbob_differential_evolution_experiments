from benchmark import run_suite, DiffEvoConfig
import cocopp

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

def population_tests(seed):
    vanilla_cfg = create_vanilla_cfg()
    folders = []

    for pop_size in [10, 25, 50, 100, 200]:
        vanilla_cfg.init_population_size = pop_size

        output_folder = f"V_pop{pop_size}"
        run_suite(vanilla_cfg, output_folder, f"V_pop{pop_size}", seed=seed, postprocess=False)
        folders.append(output_folder)

    return folders

def crossover_rate_tests(seed):
    vanilla_cfg = create_vanilla_cfg()
    folders = []

    for rate in [0.0, 0.2, 0.4, 0.5, 0.8, 1.0]:
        vanilla_cfg.crossover_rate = rate

        output_folder = f"V_cr{rate}"
        run_suite(vanilla_cfg, output_folder, f"V_cr{rate}", seed=seed, postprocess=False)
        folders.append(output_folder)

    return folders


def differential_weight_tests(seed):
    vanilla_cfg = create_vanilla_cfg()
    folders = []

    for weight in [0.2, 0.4, 0.5, 0.8, 1.0]:
        vanilla_cfg.differential_weight = weight

        output_folder = f"V_F{weight}"
        run_suite(vanilla_cfg, output_folder, f"V_F{weight}", seed=seed, postprocess=False)
        folders.append(output_folder)

    return folders

def main():
    SEED = 121
    pop_folders = population_tests(SEED)
    cr_folders = crossover_rate_tests(SEED)
    F_folders = differential_weight_tests(SEED)

    # post-process each variable test
    for folder in [pop_folders, cr_folders, F_folders]:
        cocopp.main(" ".join([exp for exp in folder]))

if __name__ == "__main__":
     main()