# USGS_MYB_Platinum.py (flowsa)
# !/usr/bin/env python3
# coding=utf-8

import io
from flowsa.common import *
from flowsa.flowbyfunctions import assign_fips_location_system
from flowsa.USGS_MYB_Common import *


"""
Projects
/
FLOWSA
/

FLOWSA-314

Import USGS Mineral Yearbook data

Description

Table T1

SourceName: USGS_MYB_Platinum 
https://www.usgs.gov/centers/nmic/platinum-group-metals-statistics-and-information

Minerals Yearbook, xls file, tab T1: SALIENT PLATINUM STATISTICS
data for:

Primary lead, refined content, domestic ores and base bullion
Palladium, Pd content:, Platinum, Pt content:

Platinum group metals; iridium
Platinum group metals; osmium
Platinum group metals; rhodium
Platinum group metals; ruthenium
There is no production value for iridium, osmium, rhodium, ruthenium
There is no export value for Osmium or Ruthenium

Years = 2014+
"""
SPAN_YEARS = "2014-2018"

def usgs_platinum_url_helper(build_url, config, args):
    """Used to substitute in components of usgs urls"""
    # URL Format, replace __year__ and __format__, either xls or xlsx.
    url = build_url
    return [url]


def usgs_platinum_call(url, usgs_response, args):
    """TODO."""
    df_raw_data = pd.io.excel.read_excel(io.BytesIO(usgs_response.content), sheet_name='T1')# .dropna()
    df_data_1 = pd.DataFrame(df_raw_data.loc[4:9]).reindex()
    df_data_1 = df_data_1.reset_index()
    del df_data_1["index"]

    df_data_2 = pd.DataFrame(df_raw_data.loc[18:30]).reindex()
    df_data_2 = df_data_2.reset_index()
    del df_data_2["index"]

    if len(df_data_1. columns) == 13:
        df_data_1.columns = ["Production", "space_6", "Units", "space_1", "year_1", "space_2", "year_2", "space_3", "year_3",
                           "space_4", "year_4", "space_5", "year_5"]
        df_data_2.columns = ["Production", "space_6", "Units", "space_1", "year_1", "space_2", "year_2", "space_3", "year_3",
                           "space_4", "year_4", "space_5", "year_5"]

    col_to_use = ["Production"]
    col_to_use.append(usgs_myb_year(SPAN_YEARS, args["year"]))
    for col in df_data_1.columns:
        if col not in col_to_use:
            del df_data_1[col]
            del df_data_2[col]

    frames = [df_data_1, df_data_2]
    df_data = pd.concat(frames)
    df_data = df_data.reset_index()
    del df_data["index"]
    return df_data


def usgs_platinum_parse(dataframe_list, args):
    """Parsing the USGS data into flowbyactivity format."""
    data = {}
    row_to_use = ["Quantity", "Palladium, Pd content", "Platinum, includes coins, Pt content", "Platinum, Pt content",
                  "Iridium, Ir content", "Osmium, Os content", "Rhodium, Rh content", "Ruthenium, Ru content",
                  "Iridium, osmium, and ruthenium, gross weight", "Rhodium, Rh content"]
    dataframe = pd.DataFrame()

    for df in dataframe_list:
        previous_name = ""
        for index, row in df.iterrows():

            if df.iloc[index]["Production"].strip() == "Exports, refined:":
                product = "exports"
            elif df.iloc[index]["Production"].strip() == "Imports for consumption, refined:":
                product = "imports"
            elif df.iloc[index]["Production"].strip() == "Mine production:2":
                product = "production"

            name_array = df.iloc[index]["Production"].strip().split(",")

            if product == "production":
                name_array = previous_name.split(",")

            previous_name = df.iloc[index]["Production"].strip()
            name = name_array[0]

            if df.iloc[index]["Production"].strip() in row_to_use:
                data = usgs_myb_static_varaibles()
                data["SourceName"] = args["source"]
                data["Year"] = str(args["year"])
                data["Unit"] = "kilograms"
                data['FlowName'] = name + " " + product
                data["Description"] = name
                data["ActivityProducedBy"] = name
                col_name = usgs_myb_year(SPAN_YEARS, args["year"])
                if str(df.iloc[index][col_name]) == "--":
                    data["FlowAmount"] = str(0)
                else:
                    data["FlowAmount"] = str(df.iloc[index][col_name])
                dataframe = dataframe.append(data, ignore_index=True)
                dataframe = assign_fips_location_system(dataframe, str(args["year"]))
    return dataframe

