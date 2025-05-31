import numpy as np
import scipy

def uniform_init(count, dimensionality, bounds=(-1.0, 1.0)):
    """Initialize population with uniform random distribution within bounds"""
    low, high = bounds
    return low + (high - low) * np.rd.random(size=(count, dimensionality))

def normal_init(count, dimensionality, bounds=(-1.0, 1.0)):
    """Initialize population with normal distribution around origin"""
    mean, std = 0.0, bounds[1]/3  # std to keep ~99.7% within bounds
    return mean + std * np.rd.randn(count, dimensionality)

def latin_hypercube_init(count, dimensionality, bounds=(-1.0, 1.0)):
    """Initialize population using Latin Hypercube sampling"""
    try:
        rng = np.random.default_rng(seed=123)
        sampler = scipy.stats.qmc.LatinHypercube(d=dimensionality, rng=rng)
        sample = sampler.random(n=count)
        low, high = bounds
        return scipy.stats.qmc.scale(sample, low, high)
    except ImportError:
        # Fallback to uniform if scipy not available
        print("Warning: scipy not available, falling back to uniform initialization")
        return uniform_init(count, dimensionality, bounds)

# Dictionary mapping strategy names to functions
init_strategies = {
    "uniform": uniform_init,
    "normal": normal_init,
    "latin_hypercube": latin_hypercube_init
}