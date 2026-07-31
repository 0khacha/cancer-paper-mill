import numpy as np

def bootstrap_ci(data, stat_fn, n_resamples=2000, ci=95):
    """Compute bootstrap CI for a given statistic."""
    data = np.array(data)
    stats = []
    n = len(data)
    np.random.seed(42)
    for _ in range(n_resamples):
        indices = np.random.randint(0, n, n)
        sample = data[indices]
        stats.append(stat_fn(sample))
    
    alpha = (100 - ci) / 2
    lower = np.percentile(stats, alpha)
    upper = np.percentile(stats, 100 - alpha)
    return lower, upper

def bootstrap_ci_diff(arr1, arr2, func, n_resamples=2000):
    np.random.seed(42)
    diffs = []
    n1 = len(arr1)
    n2 = len(arr2)
    for _ in range(n_resamples):
        samp1 = np.random.choice(arr1, size=n1, replace=True)
        samp2 = np.random.choice(arr2, size=n2, replace=True)
        diffs.append(func(samp1) - func(samp2))
    return np.percentile(diffs, [2.5, 97.5])
