import pandas as pd
import numpy as np
import os
import sys
import warnings

from handler import CEMS
from piecewiseconvex import fit_simple, fit_piecewise_convex, generate_plexos_parameters

fit_columns = [
    "min_aicc", "1_b", "1_m1", "1_x1",
    "2_b", "2_m1", "2_m2", "2_x1", "2_x2",
    "3_b", "3_m1", "3_m2", "3_m3", "3_x1", "3_x2", "2_x3"]

def linear_fits(filtered_h5, max_segments):

    results = []

    print("Opening", filtered_h5)

    with CEMS(filtered_h5, mode='r') as f:

        for group_type in f.dsets:

            print("Processing", group_type, "group...")

            result = f[group_type].unit_dfs.apply(
                lambda unit_df: unit_df.groupby("cluster").apply(
                    unit_linear_fit, max_segments=max_segments))

            result["min_aicc"].astype(int, copy=False)
            result.reset_index(2, drop=True, inplace=True)
            result["group_type"] = group_type
            result.set_index("group_type", drop=True, append=True, inplace=True)
            result = result.reorder_levels(["group_type", "unit_id", "cluster"])

            results.append(result)
            print(result)

    print("Closed", filtered_h5)
    return pd.concat(results, join="outer", sort=False)


def unit_linear_fit(unit_cluster_df, max_segments):

    load = unit_cluster_df["load"]
    heatinput = unit_cluster_df["HTINPUT"]
    minload = min(load)
    maxload = max(load)

    if ((len(np.unique(load)) < 6) or # Require a minimum amount of unique data
        (maxload - minload < 1) or # Require at least 1 MW data spread
        ("unit_id" not in unit_cluster_df.columns) or # Drop data missing certain columns
        ("cluster" not in unit_cluster_df.columns) or
        (unit_cluster_df["cluster"].iloc[0] == -1)): # Ignore the -1 (noise) clusters
        return pd.DataFrame(index=[], columns=fit_columns)

    unitname = unit_cluster_df["unit_id"].iloc[0]
    unitcluster = unit_cluster_df["cluster"].iloc[0]
    print(unitname, unitcluster, unit_cluster_df.shape,
          len(np.unique(load)), "...", end="")

    aicc1, (ms1, bs1) = fit_simple(load, heatinput)
    b1, ms1, xs1 = generate_plexos_parameters(ms1, bs1, minload, maxload)
    results1 = np.concatenate([[b1], ms1, xs1])

    aicc2, (ms2, bs2) = fit_piecewise_convex(load, heatinput, 2)
    b2, ms2, xs2 = generate_plexos_parameters(ms2, bs2, minload, maxload)
    results2 = np.concatenate([[b2], ms2, xs2])

    aicc3, (ms3, bs3) = fit_piecewise_convex(load, heatinput, 3)
    b3, ms3, xs3 = generate_plexos_parameters(ms3, bs3, minload, maxload)
    results3 = np.concatenate([[b3], ms3, xs3])

    min_aicc = np.argmin([aicc1, aicc2, aicc3]) + 1

    x = pd.DataFrame(
        np.concatenate([[min_aicc], results1, results2, results3]).reshape(1, -1),
        columns=fit_columns)
    print(" done.")
    return x


def write_group_results(df, out_dir="."):

    group_type = df.index.get_level_values("group_type")[0]
    out_path = os.path.join(out_dir, group_type + "_linpiecewise_fits.csv")
    df.to_csv(out_path)


if __name__ == "__main__":

    h5file = sys.argv[1]
    outputdir = sys.argv[2]

    # Hides some spurious numpy warnings - may want to disable if debugging.
    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore")
        #results = linear_fits(h5file, max_segments=3)

    results = linear_fits(h5file, max_segments=3)
    results.groupby(level="group_type").apply(write_group_results, out_dir=outputdir)
