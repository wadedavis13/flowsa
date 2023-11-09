# write_FBS_activity_set_BLS_QCEW.py (scripts)
# !/usr/bin/env python3
# coding=utf-8
"""
Create an activity set file file employment data. Script only needs to be
run for additional years if there are new NAICS.
"""

import pandas as pd
import flowsa.flowbyactivity
from flowsa.common import load_sector_length_cw_melt
from flowsa.settings import flowbysectoractivitysetspath

datasource = 'BLS_QCEW'
as_years = ['2002', '2010', '2011', '2012', '2013', '2014', '2015', '2016',
            '2017']

if __name__ == '__main__':

    # empty df
    df2 = pd.DataFrame()
    for y in as_years:
        df_import = flowsa.flowbyactivity.getFlowByActivity(datasource, y)

        df = (df_import[['ActivityProducedBy']]
              .drop_duplicates()
              .reset_index(drop=True)
              .rename(columns={"ActivityProducedBy": "name"})
              .assign(activity_set='qcew',
                      note=''))
        # reorder dataframe
        df = df[['activity_set', 'name', 'note']]
        # concat
        df2 = pd.concat([df2, df], ignore_index=True)
        # drop duplicates and save df
    df3 = df2.drop_duplicates()
    df3 = df3.sort_values(['activity_set', 'name']).reset_index(drop=True)

    # subset the bls qcew data to only include true NAICS
    naics_list = pd.DataFrame()
    for y in ['2012', '2017']:
        cw = load_sector_length_cw_melt(year=y)
        naics_list = pd.concat([naics_list, cw], ignore_index=True)
    naics_list = naics_list['Sector'].drop_duplicates().values.tolist()

    df4 = df3[df3['name'].isin(naics_list)]

    df4.to_csv(f"{flowbysectoractivitysetspath}/{datasource}_asets.csv",
               index=False)
