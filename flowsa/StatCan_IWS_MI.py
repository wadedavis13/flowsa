# Stat_Canada.py (flowsa)
# !/usr/bin/env python3
# coding=utf-8
'''
Pulls Statistics Canada data on water intake and discharge for 3 digit NAICS from 2005 - 2015
'''

import pandas as pd
import io
import zipfile
from flowsa.common import *

def sc_call(url, sc_response, args):
    # Convert response to dataframe
    # read all files in the stat canada zip
    with zipfile.ZipFile(io.BytesIO(sc_response.content), "r") as f:
        # read in file names
        for name in f.namelist():
            # if filename does not contain "MetaData", then create dataframe
            if "MetaData" not in name:
                data = f.open(name)
                df = pd.read_csv(data, header=0)
    return df



def sc_parse(dataframe_list, args):
    # concat dataframes
    df = pd.concat(dataframe_list, sort=True)
    # drop columns
    df = df.drop(columns=['COORDINATE', 'DECIMALS', 'DGUID', 'SYMBOL', 'TERMINATED', 'UOM_ID', 'SCALAR_ID', 'VECTOR'])
    # rename columns
    df = df.rename(columns={'GEO': 'FIPS',
                            'North American Industry Classification System (NAICS)': 'Description',
                            'REF_DATE': 'Year',
                            'STATUS': 'Spread',
                            'VALUE': "FlowAmount",
                            'Water use parameter': 'FlowName'})
    # extract NAICS as activity column. rename activity based on flowname
    df['Activity'] = df['Description'].str.extract('.*\[(.*)\].*')
    df.loc[df['Description'] == 'Total, all industries', 'Activity'] = '31-33'  # todo: change these activity names
    df.loc[df['Description'] == 'Other manufacturing industries', 'Activity'] = 'Other'
    df['FlowName'] = df['FlowName'].str.strip()
    df.loc[df['FlowName'] == 'Water intake', 'ActivityConsumedBy'] = df['Activity']
    df.loc[df['FlowName'].isin(['Water discharge', "Water recirculation"]), 'ActivityProducedBy'] = df['Activity']
    # create "unit" column
    df["Unit"] = "million " + df["UOM"] + "/year"
    # drop columns used to create unit and activity columns
    df = df.drop(columns=['SCALAR_FACTOR', 'UOM', 'Activity'])
    # Modify the assigned RSD letter values to numeric value
    df.loc[df['Spread'] == 'A', 'Spread'] = 2.5  # given range: 0.01 - 4.99%
    df.loc[df['Spread'] == 'B', 'Spread'] = 7.5  # given range: 5 - 9.99%
    df.loc[df['Spread'] == 'C', 'Spread'] = 12.5  # given range: 10 - 14.99%
    df.loc[df['Spread'] == 'D', 'Spread'] = 20  # given range: 15 - 24.99%
    df.loc[df['Spread'] == 'E', 'Spread'] = 37.5  # given range:25 - 49.99%
    df.loc[df['Spread'] == 'F', 'Spread'] = 75  # given range: > 49.99%
    df.loc[df['Spread'] == 'x', 'Spread'] = withdrawn_keyword
    # hard code data
    df['Class'] = 'Water'
    df['SourceName'] = 'StatCan_IWS_MI'
    df["MeasureofSpread"] = 'RSD'
    df["DataReliability"] = '3'
    df["DataCollection"] = '4'
    return df
