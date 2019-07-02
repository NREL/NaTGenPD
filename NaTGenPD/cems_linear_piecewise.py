import pandas as pd
import numpy as np
from re import match
import os

from handler import CEMS
from peicewiseconvex import fit_piecewise_convex, generate_plexos_parameters

blacklist=["55102_3", "7783_P001"]
#    units_keep = np.array([x not in blacklist for x in data["unit"]])
#    return data[units_keep].reset_index()

def linear_fits(filtered_h5, out_dir, max_segments=3):

    results = []

    with CEMS(filtered_h5, mode='r') as f:

        for group_type in f.dsets:

            result = f[group_type].unit_dfs.apply(
                lambda unit_df: unit_df.groupby("cluster").apply(
                    unit_linear_fit, max_segments=max_segments))

            results.append(result)

            out_path = group_type + "_linpiecewise_fits.csv"
            out_path = os.path.join(out_dir, out_path)
            result.to_csv(out_path)

    return pd.concat(results, join="inner")


def unit_linear_fit(unit_cluster_df, max_segments):

    # Ignore the -1 (noise) clusters
    if unit_cluster_id["cluster"] == -1:
        return pd.Series()

    # TODO: Fitting calculations

    return pd.Series(
        [min_aicc] + seg1_results + seg2_results + seg3_results,
        index=["min_aicc",
               "1_b", "1_m1", "1_x1",
               "2_b", "2_m1", "2_m2", "2_x1", "2_x2",
               "3_b", "3_m1", "3_m2", "3_m3", "3_x1", "3_x2", "2_x3"])


def fit(data, group_col, max_tranches):

    grouped = data.groupby(group_col)
    results = pd.DataFrame(np.nan,
                           columns=["Heat Rate Base",
                                    "Heat Rate Incr 1", "Load Point 1",
                                    "Heat Rate Incr 2", "Load Point 2",
                                    "Heat Rate Incr 3", "Load Point 3"],
                           index=[g for g, _ in grouped])
    results["Type"] = "missing"
    results["Fuel"] = "missing"
    results["Category"] = "missing"
   
    print("Fitting heat input functions...")
    for i, (groupname, group) in enumerate(grouped):

        print("  " + groupname)
        ms, bs = fit_piecewise_convex(group.load, group.HTINPUT, max_tranches)
        
        heat_rate_base, heat_rate_incrs, load_points = \
            generate_plexos_parameters(ms, bs, min(group.load), max(group.load))

        results.iloc[i, 0] = heat_rate_base

        for j in range(len(heat_rate_incrs)):
            results.iloc[i, 1 + 2*j] = heat_rate_incrs[j]
            results.iloc[i, 2 + 2*j] = load_points[j]

        results.iloc[i, 7] = group.unit_type.iloc[0]
        results.iloc[i, 8] = group.fuel_type.iloc[0]
        results.iloc[i, 9] = group.group_type.iloc[0]

    return results
    
m_idxs = np.array([1,3,5])
x_idxs = np.array([2,4,6])

def min_avg_heat_rates(row):

    base = row[0]
    ms = row[m_idxs].data
    xs = row[x_idxs].data
    dxs = np.diff(np.insert(xs, 0, 0))
    avg_hrs = (base + np.cumsum(ms * dxs)) / xs
    return min(avg_hrs)

def filter_units(data):
    
    grouped = data.groupby("Category")
    result = pd.DataFrame(np.nan, columns=data.columns, index=[])
    
    for groupname, group in grouped:
        
        if groupname in ["Boiler (Coal)", "Boiler (Natural Gas)",
                         "Combustion turbine (Natural Gas)",
                         "Combined cycle (Natural Gas)"]:
        
            # For each unit, calculate average heat rate at each load point,
            # and keep the smallest
            min_rates = group.apply(min_avg_heat_rates, axis=1, raw=True)
            mu = min_rates.mean()
            sigma = min_rates.std()
            
            if groupname in ["Boiler (Coal)", "Boiler (Natural Gas)",
                             "Combustion turbine (Natural Gas)"]:
                keepers = np.logical_and(mu - 2*sigma < min_rates, min_rates < mu + 2*sigma)
                group = group[keepers]

            else:
                keepers = np.logical_and(mu - 2*sigma < min_rates, min_rates < 9.5)
                group = group[keepers]

        result = result.append(group)
        
    return result
    
if __name__ == "__main__":
    
    # Fit individual units and do final filter
    data = load_data()
    unit_fits = fit(data, "unit", 3)
    #unit_fits = pd.read_csv("raw_unit_fits.csv", index_col=0)
    unit_fits_filtered = filter_units(unit_fits)
    unit_fits_filtered.to_csv("cems_unit_fits.csv", index_label="unit")
    
    # Use final fits to calculate generic fits
    data.index = data.unit
    data_filtered = data.loc[unit_fits_filtered.index].reset_index(drop=True)
    data = data.reset_index(drop=True)
    generic_fits_filtered = fit(data_filtered, "group_type", 1).loc[:,
                               ["Heat Rate Base", "Heat Rate Incr 1", "Load Point 1"]]
    generic_fits_filtered.to_csv("cems_generic_fits.csv", index_label="Category")

