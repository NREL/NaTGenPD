import numpy as np

def fit_simple(loads, heat_inputs):

    if len(np.unique(loads)) < 2:
        print("Unique load values:", len(np.unique(loads)))
        print(np.linalg.matrix_rank([loads, heat_inputs]))
    ms, bs = np.polyfit(loads, heat_inputs, 1)
    ms = np.array([ms])
    bs = np.array([bs])
    aicc = _fit_aicc(loads, heat_inputs,
                     np.zeros(len(loads), dtype=int), ms, bs)

    return aicc, (ms, bs)


def fit_piecewise_convex(loads, heat_rates, n_partitions,
                         n_trials=10, max_iterations=10, max_starts=10):

    best_aicc = np.inf
    best_fit = (np.full(n_partitions, np.nan), np.full(n_partitions, np.nan))

    for i in range(n_trials):

        aicc, ms, bs = _fit_piecewise_convex(
            loads, heat_rates, n_partitions, max_iterations, max_starts)

        if aicc < best_aicc:
            best_aicc = aicc
            best_fit = (ms, bs)

    return best_aicc, best_fit


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

            # Some partition spans a single load value, break and start over
            if np.any([len(np.unique(loads[partition_assignments == j])) <= 1 for j in range(n_partitions)]):
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

"""
Randomly partition load values such that each partition is guaranteed to
contain at least two distinct points (prevents rank deficiency issues in
linear fits).
"""
def _assign_random_partitions(loads, n_partitions):

    assignments = np.zeros(len(loads), dtype=int)

    if n_partitions == 1:
        return assignments

    n_bounds = n_partitions-1
    unique_loads = np.sort(np.unique(np.array(loads)))
    n_unique_loads = len(unique_loads)
    bounds = []

    while len(bounds) < n_bounds:

        validbounds = set(range(1, n_unique_loads-2))

        for i in range(n_bounds):

            if not validbounds:
                print("Infeasible partition scheme, retrying")
                bounds = []
                break

            bound = np.random.choice(list(validbounds))
            validbounds -= set(range(bound-1, bound+2))
            bounds.append(bound)

    bounds = sorted(bounds)
    bounds = unique_loads[bounds]

    for i, bound in enumerate(bounds):
        assignments[loads > bound] = i + 1

    return assignments


def _fit_partitions(loads, heat_rates, partition_assignments, ms, bs):

    for j in range(len(ms)):
        partition_idx = partition_assignments == j
        partition_loads = loads[partition_idx]
        partition_heat_rates = heat_rates[partition_idx]
        if len(np.unique(partition_loads)) < 2:
            print("Unique load values:", len(np.unique(partition_loads)))
            print("Rank:", np.linalg.matrix_rank([partition_loads, partition_heat_rates]))
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


def generate_plexos_parameters(ms, bs, min_load, max_load):

    if np.any(np.isnan(ms) | np.isnan(bs)):
        nans = np.full(ms.shape[0], np.nan)
        return np.nan, nans, nans.copy()

    # Sort lines into active partition order,
    # which is also descending y-intercept order
    ordering = np.argsort(bs)[::-1]
    ms = ms[ordering]
    bs = bs[ordering]

    load_points = (bs[1:] - bs[:-1]) / (ms[:-1] - ms[1:])
    load_points = np.append(load_points, max_load)

    # Return Base Heat Rate and lists of Heat Rate Incr, Load Point
    return bs[0], ms, load_points
