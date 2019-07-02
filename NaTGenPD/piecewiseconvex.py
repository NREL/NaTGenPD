import numpy as np
import warnings

def generate_plexos_parameters(ms, bs, min_load, max_load):

    # Sort lines into active partition order,
    # which is also descending y-intercept order
    ordering = np.argsort(bs)[::-1]
    ms = ms[ordering]
    bs = bs[ordering]

    load_points = (bs[1:] - bs[:-1]) / (ms[:-1] - ms[1:])
    load_points = np.append(load_points, max_load)

    # Return Base Heat Rate and lists of Heat Rate Incr, Load Point
    return bs[0], ms, load_points


def fit_piecewise_convex(loads, heat_rates, max_partitions,
                         n_trials=10, max_iterations=10, max_starts=10):
    best_aicc = np.inf
    best_fit = None
    
    for n_partitions in range(1, max_partitions+1):
        for i in range(n_trials):
            aicc, ms, bs = _fit_piecewise_convex(loads, heat_rates,
                                                n_partitions, max_iterations,
                                                max_starts)
            if aicc < best_aicc:
                best_aicc = aicc
                best_fit = (ms, bs)
            
    return best_fit


def _fit_piecewise_convex(loads, heat_rates, n_partitions,
                          max_iterations, max_starts):

    max_load = max(loads)
    complete = False
    ms = np.zeros(n_partitions)
    bs = np.zeros(n_partitions)
    starts = 0

    while not complete and starts <= max_starts:
        
        partition_assignments = _assign_random_partitions(loads, n_partitions)
        prev_partition_assignments = np.copy(partition_assignments)

        iterations = 0
        starts += 1
    
        while iterations < max_iterations:
            
            iterations += 1
    
            ms, bs = _fit_partitions(loads, heat_rates,
                                     prev_partition_assignments, ms, bs)
            partition_assignments = _assign_partitions(loads, ms, bs, max_load)
    
            # Lost a partition, break and start over
            if len(np.unique(partition_assignments)) < n_partitions:
                break
    
            # Converged on a local optima, break and return
            if np.array_equal(prev_partition_assignments, partition_assignments):
                complete = True
                break
    
            prev_partition_assignments = partition_assignments
            
        # Timed out, return
        if iterations == max_iterations:
            complete = True
                   
    if complete:
        aicc = _fit_aicc(loads, heat_rates, partition_assignments, ms, bs)
    else:
        aicc = np.inf

    return aicc, ms, bs


def _assign_partitions(loads, ms, bs, max_load):

    load_points = (bs[1:] - bs[:-1]) / (ms[:-1] - ms[1:])
    load_points = np.append(load_points, max_load)

    if (load_points <= 0).any() or \
       (np.diff(load_points) <= 0).any():
        return np.zeros_like(loads)

    return np.searchsorted(load_points, loads)


def _assign_random_partitions(loads, n_partitions):
    # Samples Voronoi seeds from a uniform distribution over the
    # load domain. A better-fitting distribution might be nicer...
    
    partitions = None
    
    while True:
        seed_points = np.random.uniform(min(loads), max(loads), n_partitions)
        partitions = _nearest_neighbour(loads, seed_points)
        if len(np.unique(partitions)) == n_partitions:
            break
    
    return partitions


def _nearest_neighbour(samples, prototypes):
    return np.argmin(np.absolute(np.array([samples]).T - prototypes), 1)


def _fit_partitions(loads, heat_rates, partition_assignments, ms, bs):

    with warnings.catch_warnings():
        
        warnings.simplefilter("ignore")
        
        for j in range(len(ms)):
            partition_idx = partition_assignments == j
            partition_loads = loads[partition_idx]
            partition_heat_rates = heat_rates[partition_idx]
            ms[j], bs[j] = np.polyfit(partition_loads, partition_heat_rates, 1)

    ordering = np.argsort(bs)[::-1]
    ms = ms[ordering]
    bs = bs[ordering]

    return ms, bs


def _fit_aicc(loads, heat_rates, partition_assignments, ms, bs):

    rss = 0.
    n = len(loads)
    k = 3*len(ms)

    for i, m, b in zip(range(len(ms)), ms, bs):
        partition_idx = partition_assignments == i
        partition_loads = loads[partition_idx]
        partition_heat_rates = heat_rates[partition_idx]
        rss += np.sum(((m * partition_loads + b) - partition_heat_rates)**2)

    return 2*k + n*np.log(rss) + (2*k)*(k+1) / (n - k - 1)
