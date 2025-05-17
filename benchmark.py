import cocoex
import cocopp
from differential_evolution import DiffEvoMinimizer, DiffEvoConfig

import random as rd
import numpy.random as nprd


def run_suite(cfg: DiffEvoConfig, output_folder: str = None, alg_name: str = None, suite_options: str = None, postprocess=False):
    """ Run benchmarks on DiffEvoMinimizer with given config on default bbob suite"""

    # reinitialize seeds to make each suite run reproducible independently
    rd.seed(13)
    nprd.seed(12)

    SUITE = "bbob"
    BUDGET = 1
    OPT = DiffEvoMinimizer(cfg)

    # https://numbbo.github.io/coco-doc/C/#suite-parameters
    suite = cocoex.Suite(SUITE, "", suite_options or "")

    # https://numbbo.github.io/coco-doc/C/#observer-parameters
    observer_params = ""

    if output_folder:
        observer_params += f" result_folder: {output_folder}"

    if alg_name:
        observer_params += f" algorithm_name: {alg_name}"

    observer = cocoex.Observer(SUITE, observer_params)
    repeater = cocoex.ExperimentRepeater(BUDGET)  # 0 == no repetitions
    progress_print = cocoex.utilities.MiniPrint()

    while not repeater.done():  # while budget is left and successes are few
        for problem in suite:  # loop takes 2-3 minutes x budget_multiplier
            if repeater.done(problem):
                continue  # skip this problem
            problem.observe_with(observer)  # generate data for cocopp
            problem(problem.dimension * [0])  # for better comparability
            xopt = OPT(problem, problem.dimension)

            problem(xopt)  # make sure the returned solution is evaluated
            repeater.track(problem)  # track evaluations and final_target_hit
            progress_print(problem)  # show progress

    if postprocess:
        cocopp.main(observer.result_folder + ' bfgs!')

    return observer.result_folder
