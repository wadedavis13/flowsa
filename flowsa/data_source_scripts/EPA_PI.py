# EPAN_NI.py (flowsa)
# !/usr/bin/env python3
# coding=utf-8

import io
from flowsa.flowbyfunctions import assign_fips_location_system
from flowsa.data_source_scripts.USGS_MYB_Common import *

"""
Projects
/
FLOWSA
/


Years = 2002, 2007, 2012
"""

def pi_url_helper(build_url, config, args):
    """Used to substitute in components of usgs urls"""
    url = build_url
    return [url]


def pi_call(**kwargs):
    """TODO."""
    # load arguments necessary for function
    url = kwargs['url']
    response = kwargs['r']
    args = kwargs['args']
    """Calls for the years 2002, 2007, and 2012"""
    df_legend = pd.io.excel.read_excel(io.BytesIO(response.content), sheet_name='Legend')
    df_legend = pd.DataFrame(df_legend.loc[0:18]).reindex()
    df_legend.columns = ["HUC_8", "HUC8 CODE"]
    if args['year'] == '2002':
        df_raw = pd.io.excel.read_excel(io.BytesIO(response.content), sheet_name='2002')
        df_raw = df_raw.rename(columns={'P_deposition': '2P_deposition', 'livestock_Waste_2007': 'livestock_Waste',
                                        'livestock_demand_2007': 'livestock_demand',
                                        'livestock_production_2007': 'livestock_production', '02P_Hi_P': 'P_Hi_P',
                                        'Surplus_2002': 'surplus'})
    elif args['year'] == '2007':
        df_raw = pd.io.excel.read_excel(io.BytesIO(response.content), sheet_name='2007')
        df_raw = df_raw.rename(columns={'P_deposition': '2P_deposition', 'Crop_removal_2007': 'Crop_removal',
                                        'livestock_Waste_2007': 'livestock_Waste',
                                        'livestock_demand_2007': 'livestock_demand',
                                        'livestock_production_2007': 'livestock_production', '02P_Hi_P': 'P_Hi_P',
                                        'Surplus_2007': 'surplus'})
    else:
       df_raw = pd.io.excel.read_excel(io.BytesIO(response.content), sheet_name='2012')
       df_raw = df_raw.rename(columns={'P_deposition': '2P_deposition',  'Crop_removal_2012': 'Crop_removal',
           'livestock_Waste_2012': 'livestock_Waste', 'livestock_demand_2012': 'livestock_demand',
           'livestock_production_2012': 'livestock_production', '02P_Hi_P': 'P_Hi_P',
           'Surplus_2012': 'surplus'})

    for col_name in df_raw.columns:
        for i in range(len(df_legend)):
            if '_20' in df_legend.loc[i, "HUC_8"]:
                legend_str = str(df_legend.loc[i, "HUC_8"])
                list = legend_str.split('_20')
                df_legend.loc[i, "HUC_8"] = list[0]

            if col_name == df_legend.loc[i, "HUC_8"]:
                df_raw = df_raw.rename(columns={col_name: df_legend.loc[i, "HUC8 CODE"]})
    df_des = df_raw.filter(['HUC8 CODE', 'State Name'])
    df_raw = df_raw.drop(columns=['State Name', 'State FP Code'])

    # use "melt" fxn to convert colummns into rows
    df = df_raw.melt(id_vars=["HUC8 CODE"],
                          var_name="ActivityProducedBy",
                          value_name="FlowAmount")

    df = df.merge(df_des, left_on='HUC8 CODE', right_on='HUC8 CODE')
    df = df.rename(columns={"HUC8 CODE": "Location"})
    df = df.rename(columns={"State Name": "Description"})
    return df


def pi_parse(dataframe_list, args):
    """Parsing the USGS data into flowbyactivity format."""
    data = {}
    row_to_use = ["Production2", "Production", "Imports for consumption"]
    df = pd.DataFrame()



    for df in dataframe_list:
        for i in range(len(df)):
            apb = df.loc[i, "ActivityProducedBy"]
            apb_str = str(apb)
            if '(' in apb_str:
                apb_split = apb_str.split('(')
                activity = apb_split[0].strip()
                unit_str = apb_split[1]
                unit_list = unit_str.split(')')
                unit = unit_list[0]
                df.loc[i, "ActivityProducedBy"] = activity
                df.loc[i, "ActivityConsumedBy"] = None
                df.loc[i, "Units"] = unit
                if activity == 'Livestock Waste recovered and applied to fields':
                    df.loc[i, "ActivityProducedBy"] = None
                    df.loc[i, "ActivityConsumedBy"] = activity
            else:
                df.loc[i, "Units"] = None
        df["Compartment "] = "ground"
        df["Class"] = "Chemicals"
        df["SourceName"] = "EPA_NI"
        df["LocationSystem"] = 'HUC'
        df["Year"] = str(args["year"])
        df["FlowName"] = 'Phosphorus'

    return df

