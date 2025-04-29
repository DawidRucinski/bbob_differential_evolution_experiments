import cocoex  # experimentation module
import cocopp  # post-processing module (not strictly necessary)
from differential_evolution import DiffEvoMinimizer, DiffEvoConfig

### input
suite_name = "bbob"

cfg = DiffEvoConfig()
cfg.init_population_size = 50
cfg.replaced_count = 15
cfg.replacement_strategy = "random"

opt = DiffEvoMinimizer(cfg)

budget_multiplier = 1  # x dimension, increase to 3, 10, 30,...

### prepare
suite = cocoex.Suite(suite_name, "", "")  # see https://numbbo.github.io/coco-doc/C/#suite-parameters
output_folder = '{}_{}D_on_{}'.format(cfg.short_repr(), int(budget_multiplier), suite_name)
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
cocopp.main(observer.result_folder + ' bfgs!');  # re-run folders look like "...-001" etc