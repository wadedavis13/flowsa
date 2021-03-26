# USGS_MYB_Lead.py (flowsa)
# !/usr/bin/env python3
# coding=utf-8

import io
from flowsa.flowbyfunctions import assign_fips_location_system
from flowsa.data_source_scripts.USGS_MYB_Common import *

"""
SourceName: USGS_MYB_Lead
https://www.usgs.gov/centers/nmic/lead-statistics-and-information

Minerals Yearbook, xls file, tab T1: SALIENT LEAD STATISTICS
data for:

Primary lead, refined content, domestic ores and base bullion
Secondary lead, lead content

Years = 2010+
"""
SPAN_YEARS = "2012-2016"

def usgs_lead_url_helper(build_url, config, args):
    """Used to substitute in components of usgs urls"""
    url = build_url
    return [url]


def usgs_lead_call(url, usgs_response, args):
    """TODO."""
    df_raw_data = pd.io.excel.read_excel(io.BytesIO(usgs_response.content), sheet_name='T1')# .dropna()
    df_data = pd.DataFrame(df_raw_data.loc[8:15]).reindex()
    df_data = df_data.reset_index()
    del df_data["index"]

    if len(df_data.columns) > 12:
        for x in range(12, len(df_data.columns)):
            col_name = "Unnamed: " + str(x)
            del df_data[col_name]

    if len(df_data. columns) == 12:
        df_data.columns = ["Production", "Units", "space_1", "year_1", "space_2", "year_2", "space_3", "year_3",
                           "space_4", "year_4", "space_5", "year_5"]

    col_to_use = ["Production", "Units"]
    col_to_use.append(usgs_myb_year(SPAN_YEARS, args["year"]))
    for col in df_data.columns:
        if col not in col_to_use:
            del df_data[col]

    return df_data


def usgs_lead_parse(dataframe_list, args):
    """Parsing the USGS data into flowbyactivity format."""
    data = {}
    name = usgs_myb_name(args["source"])
    des = name
    row_to_use = ["Primary lead, refined content, domestic ores and base bullion", "Secondary lead, lead content",
                   "Lead ore and concentrates", "Lead in base bullion"]
    import_export = ["Exports, lead content:","Imports for consumption, lead content:"]
    dataframe = pd.DataFrame()
    product = "production"
    for df in dataframe_list:
        for index, row in df.iterrows():
            if df.iloc[index]["Production"].strip() in import_export:
                if df.iloc[index]["Production"].strip() == "Exports, lead content:":
                    product = "exports"
                elif df.iloc[index]["Production"].strip() == "Imports for consumption, lead content:":
                    product = "imports"
            if df.iloc[index]["Production"].strip() in row_to_use:
                data = usgs_myb_static_varaibles()
                data["SourceName"] = args["source"]
                data["Year"] = str(args["year"])
                data["Unit"] = "Metric Tons"
                data['FlowName'] = name + " " + product + " " + df.iloc[index]["Production"]
                data["ActivityProducedBy"] = df.iloc[index]["Production"]
                col_name = usgs_myb_year(SPAN_YEARS, args["year"])
                if str(df.iloc[index][col_name]) == "--":
                    data["FlowAmount"] = str(0)
                else:
                    data["FlowAmount"] = str(df.iloc[index][col_name])
                dataframe = dataframe.append(data, ignore_index=True)
                dataframe = assign_fips_location_system(dataframe, str(args["year"]))
    return dataframe

