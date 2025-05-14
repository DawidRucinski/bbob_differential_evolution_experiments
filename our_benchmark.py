import cocoex  # experimentation module
import cocopp  # post-processing module (not strictly necessary)
from differential_evolution import DiffEvoMinimizer, DiffEvoConfig

import random as rd
import numpy.random as nprd

def run_suite(cfg, output_folder=None):
    rd.seed(13)
    nprd.seed(12)  

    suite_name = "bbob"
    
    opt = DiffEvoMinimizer(cfg)
    budget_multiplier = 1  # x dimension, increase to 3, 10, 30,...

    ### prepare
    suite = cocoex.Suite(suite_name, "", "")  # see https://numbbo.github.io/coco-doc/C/#suite-parameters

    observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
    repeater = cocoex.ExperimentRepeater(budget_multiplier)  # 0 == no repetitions
    minimal_print = cocoex.utilities.MiniPrint()

    ### go
    while not repeater.done():  # while budget is left and successes are few
        for problem in suite:  # loop takes 2-3 minutes x budget_multiplier
            if repeater.done(problem):
                continue  # skip this problem
            problem.observe_with(observer)  # generate data for cocopp
            problem(problem.dimension * [0])  # for better comparability
            xopt = opt(problem, problem.dimension)

            problem(xopt)  # make sure the returned solution is evaluated
            repeater.track(problem)  # track evaluations and final_target_hit
            minimal_print(problem)  # show progress

    ### post-process data 
    # cocopp.main(observer.result_folder + ' bfgs!');  # re-run folders look like "...-001" etc


def create_vanilla_cfg():
    cfg = DiffEvoConfig()
    cfg.replaced_count = 0
    cfg.replacement_strategy = ""
    
    cfg.min_generations_before_convergence = 5
    cfg.tolerance = 1e-8

    cfg.init_population_size = 100
    cfg.differential_weight = 0.4
    cfg.crossover_rate = 0.9

    return cfg

def population_tests():
    vanilla_cfg = create_vanilla_cfg()

    for pop_size in [25, 50, 100, 200]:
        vanilla_cfg.init_population_size = pop_size
        run_suite(vanilla_cfg, f"V_pop{pop_size}")


def crossover_rate_tests():
    vanilla_cfg = create_vanilla_cfg()

    for rate in [0.2, 0.4, 0.5, 0.8, 1.0]:
        vanilla_cfg.crossover_rate = rate
        run_suite(vanilla_cfg, f"V_cr{rate}")

def differential_weight_tests():
    vanilla_cfg = create_vanilla_cfg()

    for weight in [0.2, 0.4, 0.5, 0.8, 1.0]:
        vanilla_cfg.differential_weight = weight
        run_suite(vanilla_cfg, f"V_F{weight}")


def main():
    population_tests()
    crossover_rate_tests()
    differential_weight_tests()


if __name__ == "__main__":
     main()