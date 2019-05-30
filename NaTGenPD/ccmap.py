import numpy as np
import pandas as pd
import re

def method_1(boilers, eia_plants):

    # Create boiler-specific unit (Method 1)
    no_eia_plant = boilers.loc[~np.in1d(boilers["Plant Code"], eia_plants), :].reset_index()
    no_eia_plant["Unit Code"] = no_eia_plant["Boiler ID"]
    no_eia_plant["Unit Code Method"] = 1

    return no_eia_plant

def method_2_3(boilers23, boilers_generators, generators):

    # Get boiler -> generator matches (Methods 2 + 3)
    boilers_units = boilers23 \
             .join(boilers_generators, on=["Plant Code", "Boiler ID"], how="inner") \
             .join(generators[["Unit Code"]], on=["Plant Code", "Generator ID"], how="inner") \
             .reset_index().drop_duplicates(["CEMSUnit", "Unit Code"])

    gen_missing_unit_code = boilers_units["Unit Code"].isna()

    # Assign unit code directly (Method 2)
    direct_result = boilers_units.loc[~gen_missing_unit_code, :].copy()
    direct_result["Unit Code Method"] = 2

    # Create generator-specific unit (Method 3)
    direct_nounit_result = boilers_units.loc[gen_missing_unit_code, :].copy()
    direct_nounit_result["Unit Code"] = direct_nounit_result["Generator ID"]
    direct_nounit_result["Unit Code Method"] = 3

    return direct_result, direct_nounit_result


def method_4(boilers4567, generators_cc):

    # Check for no CA/CTs
    boilers_plants = boilers4567.loc[
        ~np.in1d(boilers4567["Plant Code"], generators_cc["Plant Code"]), :].copy()

    # Create boiler-specific unit (Method 4)
    boilers_plants["Unit Code"] = boilers_plants["Boiler ID"].astype(str)
    boilers_plants["Unit Code Method"] = 4

    return boilers_plants.reset_index()

def method_5(boilers4567, generators_cc):

    # Check for single unit code among all CA/CTs in plant
    plants_units = generators_cc.loc[
        np.in1d(generators_cc["Plant Code"], boilers4567["Plant Code"]),
        ["Plant Code", "Unit Code"]].drop_duplicates().set_index("Plant Code")["Unit Code"]
    unit_code_count = plants_units.groupby(level="Plant Code").nunique()
    single_unit_plants = unit_code_count.loc[unit_code_count == 1].index.get_values()

    # Assign all boilers in plant to same unit code if single unit code exists (Method 5)
    single_unit_plants = plants_units.loc[single_unit_plants]
    result = boilers4567.join(single_unit_plants, on="Plant Code", how="right").reset_index()
    result["Unit Code Method"] = 5

    return result


def method_6_7(boilers4567, generators_cc):

    # Check for nonsingle unit code among all CA/CTs in plant
    plants_units = generators_cc.loc[
        np.in1d(generators_cc["Plant Code"], boilers4567["Plant Code"]),
        ["Plant Code", "Unit Code"]].drop_duplicates().set_index("Plant Code")["Unit Code"]
    unit_code_count = plants_units.groupby(level="Plant Code").nunique()
    nonsingle_unit_plants = unit_code_count.loc[unit_code_count != 1].index.get_values()

    # Group boilers and generators by plant
    boiler_groups = boilers4567.loc[
        np.in1d(boilers4567["Plant Code"], nonsingle_unit_plants),
        :].reset_index().groupby("Plant Code")
    gen_groups = generators_cc.loc[
        generators_cc["Prime Mover"] == "CT", :].groupby("Plant Code")

    colnames = ["Plant Code", "Boiler ID", "Generator ID", "Unit Code"]
    result6 = pd.DataFrame(columns=colnames)
    result7 = pd.DataFrame(columns=colnames)

    # Match boilers and generators by sorting
    for plant in nonsingle_unit_plants:

        bs = boiler_groups.get_group(plant).sort_values("Boiler ID")
        gs = gen_groups.get_group(plant).sort_values("Generator ID")

        n_bs = len(bs.index)
        n_gs = len(gs.index)

        # Match boilers to generator unit codes (Method 6)
        if n_bs <= n_gs: 
            gs = gs.head(n_bs)
            result6 = result6.append(pd.DataFrame({
                "CEMSUnit": np.array(bs["CEMSUnit"]),
                "Plant Code": plant,
                "Boiler ID": np.array(bs["Boiler ID"]),
                "Generator ID": np.array(gs["Generator ID"]),
                "Unit Code": np.array(gs["Unit Code"])}), sort=True)

        # Match boilers to generator unit codes,
        # creating new units for extra boilers (Method 7)
        else:
            bs_rem = bs.tail(n_bs - n_gs)
            bs = bs.head(n_gs)
            result7 = result7.append(pd.DataFrame({
                "CEMSUnit": np.array(bs["CEMSUnit"]),
                "Plant Code": plant,
                "Boiler ID": np.array(bs["Boiler ID"]),
                "Generator ID": np.array(gs["Generator ID"]),
                "Unit Code": np.array(gs["Unit Code"])}), sort=True
                ).append(pd.DataFrame({
                "CEMSUnit": np.array(bs_rem["CEMSUnit"]),
                "Plant Code": plant,
                "Boiler ID": np.array(bs_rem["Boiler ID"]),
                "Unit Code": np.array(bs_rem["Boiler ID"])}), sort=True)

    result6["Unit Code Method"] = 6
    result7["Unit Code Method"] = 7

    return result6, result7


if __name__ == "__main__":

    # Load CEMS boilers
    boilers = pd.read_csv("../bin/emission_01-17-2017.csv",
        usecols=[2,3,25], header=0, names=["Plant Code", "Boiler ID", "Unit Type"])
    boilers = boilers.loc[["combined cycle" in ut.lower() for ut in boilers["Unit Type"]] , :]
    boilers.drop("Unit Type", axis=1, inplace=True)
    boilers.index = boilers["Plant Code"].astype(str) + "_" + boilers["Boiler ID"]
    boilers.index.name = "CEMSUnit"

    # Load boiler-generator mapping
    boilers_generators = pd.read_excel(
        "../bin/6_1_EnviroAssoc_Y2017.xlsx", "Boiler Generator",
        header=1, usecols=[2,4,5],
        index_col=[0,1], skipfooter=1)

    def read_generators(f, sheet):
        return f.parse(sheet, header=1, usecols=[2,6,8,9],
                       index_col=[0,1], skipfooter=1)

    # Load generator-unit mapping
    with pd.ExcelFile("../bin/3_1_Generator_Y2017.xlsx") as f:
        generators = read_generators(f, "Operable")
        generators_retired = read_generators(f, "Retired and Canceled")
        generators_proposed = read_generators(f, "Proposed")

    generators = generators.append(generators_retired, sort=True).append(generators_proposed, sort=True)
    generators_cc = generators.loc[np.in1d(generators["Prime Mover"], ["CA", "CT"]), :].reset_index()

    # Any CC CA/CTs without a unit code are assigned to a plant-wide unit
    gcc_nounitcode = generators_cc["Unit Code"].isna()
    generators_cc.loc[gcc_nounitcode, "Unit Code"] = ""

    eia_plants = [p for (p, g) in generators.index]
    eia_plants_boilers = list(boilers_generators.index)

    boilers_234567 = boilers.loc[np.in1d(boilers["Plant Code"], eia_plants), :]
    boilers_23 = np.array([(p, b) in eia_plants_boilers
                           for (_, p, b) in boilers_234567.itertuples()])
    boilers_4567 = boilers_234567.loc[~boilers_23, :]
    boilers_23 = boilers_234567.loc[boilers_23, :]

    # Process CEMS boiler + EIA860 plant/generator data
    result1          = method_1(boilers, eia_plants)
    result2, result3 = method_2_3(boilers_23, boilers_generators, generators)
    result4          = method_4(boilers_4567, generators_cc)
    result5          = method_5(boilers_4567, generators_cc)
    result6, result7 = method_6_7(boilers_4567, generators_cc)

    # Recombine results from all methods
    result = pd.concat([result1, result2, result3, result4,
                        result5, result6, result7], sort=True).set_index("CEMSUnit")[
                        ["Plant Code", "Boiler ID", "Generator ID", "Unit Code", "Unit Code Method"]]
    result["CCUnit"] = result["Plant Code"].astype(str) + "_" + result["Unit Code"].astype(str)

    print(str(len(boilers.index)) + " input boilers")
    print(str(len(result.index)) + " output mappings")
    print(str(len(result.drop_duplicates(["Plant Code", "Boiler ID"]).index)) + " unique output mappings")
    print(result.groupby("Unit Code Method").count())

    result.to_csv("../bin/cems_cc_mapping.csv")
