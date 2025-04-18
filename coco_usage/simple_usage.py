import cocoex  # experimentation module
import scipy  # to define the solver to be benchmarked

### input
suite_name = "bbob"
fmin = scipy.optimize.fmin  # optimizer to be benchmarked

### prepare
suite = cocoex.Suite(suite_name, "", "")
observer = cocoex.Observer(suite_name, "")

### go
for problem in suite:  # this loop may take several minutes or more
    problem.observe_with(observer)  # generates the data for cocopp
    fmin(problem, problem.initial_solution, disp=False)