# write_Crosswalk_Blackhurst.py (scripts)
# !/usr/bin/env python3
# coding=utf-8

"""
Create a crosswalk linking Blackhurst IO vectors to NAICS

"""
import pandas as pd
from flowsa.common import load_crosswalk
from flowsa.settings import datapath
from scripts.common_scripts import unique_activity_names, order_crosswalk


def assign_naics(df):
    """
    Function to assign NAICS codes to each dataframe activity
    :param df: df, a FlowByActivity subset that contains unique activity names
    :return: df with assigned Sector columns
    """

    cw_load = load_crosswalk('BEA')
    cw = cw_load[['BEA_2012_Detail_Code',
                  'NAICS_2012_Code']].drop_duplicates().reset_index(drop=True)
    # least aggregate level that applies is 5 digits
    cw = cw[cw['NAICS_2012_Code'].apply(lambda x: len(str(x)) == 6)].reset_index(drop=True)

    cw = cw.sort_values(['BEA_2012_Detail_Code', 'NAICS_2012_Code'])

    df = pd.merge(df, cw, left_on='Activity', right_on='BEA_2012_Detail_Code')
    df = df.drop(columns=["BEA_2012_Detail_Code"])
    df = df.rename(columns={"NAICS_2012_Code": "Sector"})
    df['SectorSourceName'] = 'NAICS_2012_Code'

    return df


if __name__ == '__main__':
    # select years to pull unique activity names
    year = '2002'
    # datasource
    datasource = 'Blackhurst_IO'
    # df of unique ers activity names
    df = unique_activity_names(datasource, year)
    # add manual naics 2012 assignments
    df = assign_naics(df)
    # drop any rows where naics12 is 'nan'
    # (because level of detail not needed or to prevent double counting)
    df.dropna(subset=["Sector"], inplace=True)
    # assign sector type
    df['SectorType'] = None
    # sort df
    df = order_crosswalk(df)
    # save as csv
    df.to_csv(datapath + "activitytosectormapping/" +
              "Crosswalk_" + datasource + "_toNAICS.csv", index=False)
