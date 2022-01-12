# USDA_CoA_Cropland.py (flowsa)
# !/usr/bin/env python3
# coding=utf-8

"""
Functions used to import and parse USDA Census of Ag Cropland data
"""

import json
import numpy as np
import pandas as pd
from flowsa.common import US_FIPS, abbrev_us_state, WITHDRAWN_KEYWORD
from flowsa.data_source_scripts.USDA_CoA_Cropland_NAICS import \
    disaggregate_pastureland, disaggregate_cropland
from flowsa.flowbyfunctions import assign_fips_location_system


def CoA_Cropland_URL_helper(*, build_url, config, **_):
    """
    This helper function uses the "build_url" input from flowbyactivity.py,
    which is a base url for data imports that requires parts of the url text
    string to be replaced with info specific to the data year. This function
    does not parse the data, only modifies the urls from which data is
    obtained.
    :param build_url: string, base url
    :param config: dictionary, items in FBA method yaml
    :return: list, urls to call, concat, parse, format into Flow-By-Activity
        format
    """
    # initiate url list for coa cropland data
    urls = []

    # call on state acronyms from common.py (and remove entry for DC)
    state_abbrevs = abbrev_us_state
    state_abbrevs = {k: v for (k, v) in state_abbrevs.items() if k != "DC"}

    # replace "__aggLevel__" in build_url to create three urls
    for x in config['agg_levels']:
        for y in config['sector_levels']:
            # at national level, remove the text string calling for
            # state acronyms
            if x == 'NATIONAL':
                url = build_url
                url = url.replace("__aggLevel__", x)
                url = url.replace("__secLevel__", y)
                url = url.replace("&state_alpha=__stateAlpha__", "")
                if y == "ECONOMICS":
                    url = url.replace(
                        "AREA%20HARVESTED&statisticcat_desc=AREA%20IN%20"
                        "PRODUCTION&statisticcat_desc=TOTAL&statisticcat_desc="
                        "AREA%20BEARING%20%26%20NON-BEARING",
                        "AREA&statisticcat_desc=AREA%20OPERATED")
                else:
                    url = url.replace("&commodity_desc=AG%20LAND&"
                                      "commodity_desc=FARM%20OPERATIONS", "")
                urls.append(url)
            else:
                # substitute in state acronyms for state and county url calls
                for z in state_abbrevs:
                    url = build_url
                    url = url.replace("__aggLevel__", x)
                    url = url.replace("__secLevel__", y)
                    url = url.replace("__stateAlpha__", z)
                    if y == "ECONOMICS":
                        url = url.replace(
                            "AREA%20HARVESTED&statisticcat_desc=AREA%20IN%20"
                            "PRODUCTION&statisticcat_desc=TOTAL&"
                            "statisticcat_desc=AREA%20BEARING%20%26%20NON-BEARING",
                            "AREA&statisticcat_desc=AREA%20OPERATED")
                    else:
                        url = url.replace("&commodity_desc=AG%20LAND&commodity_"
                                          "desc=FARM%20OPERATIONS", "")
                    urls.append(url)
    return urls


def coa_cropland_call(*, resp, **_):
    """
    Convert response for calling url to pandas dataframe, begin parsing df
    into FBA format
    :param url: string, url
    :param resp: df, response from url call
    :param args: dictionary, arguments specified when running
        flowbyactivity.py ('year' and 'source')
    :return: pandas dataframe of original source data
    """
    cropland_json = json.loads(resp.text)
    df_cropland = pd.DataFrame(data=cropland_json["data"])
    return df_cropland


def coa_cropland_parse(*, df_list, year, **_):
    """
    Combine, parse, and format the provided dataframes
    :param df_list: list of dataframes to concat and format
    :param args: dictionary, used to run flowbyactivity.py
        ('year' and 'source')
    :return: df, parsed and partially formatted to flowbyactivity
        specifications
    """
    df = pd.concat(df_list, sort=False)
    # specify desired data based on domain_desc
    df = df[~df['domain_desc'].isin(
        ['ECONOMIC CLASS', 'FARM SALES', 'IRRIGATION STATUS', 'CONCENTRATION',
         'ORGANIC STATUS', 'NAICS CLASSIFICATION', 'PRODUCERS'])]
    df = df[df['statisticcat_desc'].isin(
        ['AREA HARVESTED', 'AREA IN PRODUCTION', 'AREA BEARING & NON-BEARING',
         'AREA', 'AREA OPERATED', 'AREA GROWN'])]
    # drop rows that subset data into farm sizes (ex. 'area harvested:
    # (1,000 to 1,999 acres)
    df = df[~df['domaincat_desc'].str.contains(
        ' ACRES')].reset_index(drop=True)
    # drop Descriptions that contain certain phrases, as these data are
    # included in other categories
    df = df[~df['short_desc'].str.contains(
        'FRESH MARKET|PROCESSING|ENTIRE CROP|NONE OF CROP|PART OF CROP')]
    # drop Descriptions that contain certain phrases - only occur in
    # AG LAND data
    df = df[~df['short_desc'].str.contains(
        'INSURANCE|OWNED|RENTED|FAILED|FALLOW|IDLE')].reset_index(drop=True)
    # Many crops are listed as their own commodities as well as grouped
    # within a broader category (for example, orange
    # trees are also part of orchards). As this dta is not needed,
    # takes up space, and can lead to double counting if
    # included, want to drop these unused columns
    # subset dataframe into the 5 crop types and land in farms and drop rows
    # crop totals: drop all data
    # field crops: don't want certain commodities and don't
    # want detailed types of wheat, cotton, or sunflower
    df_fc = df[df['group_desc'] == 'FIELD CROPS']
    df_fc = df_fc[~df_fc['commodity_desc'].isin(
        ['GRASSES', 'GRASSES & LEGUMES, OTHER', 'LEGUMES', 'HAY', 'HAYLAGE'])]
    df_fc = df_fc[~df_fc['class_desc'].str.contains(
        'SPRING|WINTER|TRADITIONAL|OIL|PIMA|UPLAND', regex=True)]
    # fruit and tree nuts: only want a few commodities
    df_ftn = df[df['group_desc'] == 'FRUIT & TREE NUTS']
    df_ftn = df_ftn[df_ftn['commodity_desc'].isin(
        ['BERRY TOTALS', 'ORCHARDS'])]
    df_ftn = df_ftn[df_ftn['class_desc'].isin(['ALL CLASSES'])]
    # horticulture: only want a few commodities
    df_h = df[df['group_desc'] == 'HORTICULTURE']
    df_h = df_h[df_h['commodity_desc'].isin(
        ['CUT CHRISTMAS TREES', 'SHORT TERM WOODY CROPS'])]
    # vegetables: only want a few commodities
    df_v = df[df['group_desc'] == 'VEGETABLES']
    df_v = df_v[df_v['commodity_desc'].isin(['VEGETABLE TOTALS'])]
    # only want ag land and farm operations in farms & land & assets
    df_fla = df[df['group_desc'] == 'FARMS & LAND & ASSETS']
    df_fla = df_fla[df_fla['short_desc'].str.contains(
        "AG LAND|FARM OPERATIONS")]
    # drop the irrigated acreage in farms (want the irrigated harvested acres)
    df_fla = df_fla[
        ~((df_fla['domaincat_desc'] == 'AREA CROPLAND, HARVESTED: (ANY)') &
          (df_fla['domain_desc'] == 'AREA CROPLAND, HARVESTED') &
          (df_fla['short_desc'] == 'AG LAND, IRRIGATED - ACRES'))]
    # concat data frames
    df = pd.concat([df_fc, df_ftn, df_h, df_v, df_fla],
                   sort=False).reset_index(drop=True)
    # drop unused columns
    df = df.drop(columns=['agg_level_desc', 'location_desc', 'state_alpha',
                          'sector_desc', 'country_code', 'begin_code',
                          'watershed_code', 'reference_period_desc',
                          'asd_desc', 'county_name', 'source_desc',
                          'congr_district_code', 'asd_code', 'week_ending',
                          'freq_desc', 'load_time', 'zip_5', 'watershed_desc',
                          'region_desc', 'state_ansi', 'state_name',
                          'country_name', 'county_ansi', 'end_code',
                          'group_desc'])
    # create FIPS column by combining existing columns
    df.loc[df['county_code'] == '', 'county_code'] = '000'
    df['Location'] = df['state_fips_code'] + df['county_code']
    df.loc[df['Location'] == '99000', 'Location'] = US_FIPS

    # address non-NAICS classification data
    # use info from other columns to determine flow name
    df.loc[:, 'FlowName'] = df['statisticcat_desc'] + ', ' + \
                            df['prodn_practice_desc']
    df.loc[:, 'FlowName'] = df['FlowName'].str.replace(
        ", ALL PRODUCTION PRACTICES", "", regex=True)
    df.loc[:, 'FlowName'] = df['FlowName'].str.replace(
        ", IN THE OPEN", "", regex=True)
    # combine column information to create activity
    # information, and create two new columns for activities
    df['Activity'] = df['commodity_desc'] + ', ' + df['class_desc'] + ', ' + \
                     df['util_practice_desc']  # drop this column later
    # not interested in all data from class_desc
    df['Activity'] = df['Activity'].str.replace(
        ", ALL CLASSES", "", regex=True)
    # not interested in all data from class_desc
    df['Activity'] = df['Activity'].str.replace(
        ", ALL UTILIZATION PRACTICES", "", regex=True)
    df['ActivityProducedBy'] = np.where(
        df["unit_desc"] == 'OPERATIONS', df["Activity"], None)
    df['ActivityConsumedBy'] = np.where(
        df["unit_desc"] == 'ACRES', df["Activity"], None)

    # rename columns to match flowbyactivity format
    df = df.rename(columns={"Value": "FlowAmount", "unit_desc": "Unit",
                            "year": "Year", "CV (%)": "Spread",
                            "short_desc": "Description"})
    # drop remaining unused columns
    df = df.drop(columns=['Activity', 'class_desc', 'commodity_desc',
                          'domain_desc', 'state_fips_code', 'county_code',
                          'statisticcat_desc', 'prodn_practice_desc',
                          'domaincat_desc', 'util_practice_desc'])
    # modify contents of units column
    df.loc[df['Unit'] == 'OPERATIONS', 'Unit'] = 'p'
    # modify contents of flowamount column, "D" is supressed data,
    # "z" means less than half the unit is shown
    df['FlowAmount'] = df['FlowAmount'].str.strip()  # trim whitespace
    df.loc[df['FlowAmount'] == "(D)", 'FlowAmount'] = WITHDRAWN_KEYWORD
    df.loc[df['FlowAmount'] == "(Z)", 'FlowAmount'] = WITHDRAWN_KEYWORD
    df['FlowAmount'] = df['FlowAmount'].str.replace(",", "", regex=True)
    # USDA CoA 2017 states that (H) means CV >= 99.95,
    # therefore replacing with 99.95 so can convert column to int
    # (L) is a CV of <= 0.05
    df['Spread'] = df['Spread'].str.strip()
    df.loc[df['Spread'] == "(H)", 'Spread'] = 99.95
    df.loc[df['Spread'] == "(L)", 'Spread'] = 0.05
    df.loc[df['Spread'] == "", 'Spread'] = None
    df.loc[df['Spread'] == "(D)", 'Spread'] = WITHDRAWN_KEYWORD
    # add location system based on year of data
    df = assign_fips_location_system(df, year)
    # Add hardcoded data
    df['Class'] = np.where(df["Unit"] == 'ACRES', "Land", "Other")
    df['SourceName'] = "USDA_CoA_Cropland"
    df['FlowType'] = 'ELEMENTARY_FLOW'
    df['MeasureofSpread'] = "RSD"
    df['DataReliability'] = 5  # tmp
    df['DataCollection'] = 2

    return df


def coa_irrigated_cropland_fba_cleanup(fba, **kwargs):
    """
    When using irrigated cropland, aggregate sectors to cropland and total
    ag land. Doing this because published values for irrigated harvested
    cropland do not include the water use for vegetables, woody crops, berries.
    :param fba: df, COA FBA format
    :return: df, COA with dropped rows based on ActivityConsumedBy column
    """

    fba =\
        fba[~fba['ActivityConsumedBy'].isin(['AG LAND',
                                             'AG LAND, CROPLAND, HARVESTED']
                                            )].reset_index(drop=True)

    return fba


def coa_nonirrigated_cropland_fba_cleanup(fba, **kwargs):
    """
    When using irrigated cropland, aggregate sectors to cropland and total
    ag land. Doing this because published values for irrigated harvested
    cropland do not include the water use for vegetables, woody crops, berries.
    :param fba: df, COA when using non-irrigated data
    :return: df, COA nonirrigated data, modified
    """

    # drop rows of data that contain certain strings
    fba = fba[~fba['ActivityConsumedBy'].isin(
        ['AG LAND', 'AG LAND, CROPLAND, HARVESTED'])]

    # when include 'area harvested' and 'area in production' in
    # single dataframe, which is necessary to include woody crops,
    # 'vegetable totals' are double counted
    fba = fba[~((fba['FlowName'] == 'AREA IN PRODUCTION') &
                (fba['ActivityConsumedBy'] == 'VEGETABLE TOTALS'))]

    return fba


def disaggregate_coa_cropland_to_6_digit_naics(
        fba_w_sector, attr, method, **kwargs):
    """
    Disaggregate usda coa cropland to naics 6
    :param fba_w_sector: df, CoA cropland data, FBA format with sector columns
    :param attr: dictionary, attribute data from method yaml for activity set
    :param method: dictionary, FBS method yaml
    :param kwargs: dictionary, arguments that might be required for other
        functions. Currently includes data source name.
    :return: df, CoA cropland with disaggregated NAICS sectors
    """

    # define the activity and sector columns to base modifications on
    # these definitions will vary dependent on class type
    activity_col = 'ActivityConsumedBy'
    sector_col = 'SectorConsumedBy'

    # drop rows without assigned sectors
    fba_w_sector = fba_w_sector[
        ~fba_w_sector[sector_col].isna()].reset_index(drop=True)

    # modify the flowamounts related to the 6 naics 'orchards' are mapped to
    fba_w_sector = modify_orchard_flowamounts(
        fba_w_sector, activity_column=activity_col)

    # use ratios of usda 'land in farms' to determine animal use of
    # pasturelands at 6 digit naics
    fba_w_sector = disaggregate_pastureland(
        fba_w_sector, attr, method, year=attr['allocation_source_year'],
        sector_column=sector_col,
        download_FBA_if_missing=kwargs['download_FBA_if_missing'])

    # use ratios of usda 'harvested cropland' to determine missing 6 digit naics
    fba_w_sector = disaggregate_cropland(fba_w_sector, attr,
                                         method, year=attr['allocation_source_year'],
                                         sector_column=sector_col,
                                         download_FBA_if_missing=kwargs['download_FBA_if_missing'])

    return fba_w_sector


def coa_cropland_w_naics_cleanup(df_w_sec, **kwargs):
    # define the activity and sector columns to base modifications on
    # these definitions will vary dependent on class type
    activity_col = 'ActivityConsumedBy'
    sector_col = 'SectorConsumedBy'

    # drop rows without assigned sectors
    fba_w_sector = df_w_sec[~df_w_sec[sector_col].isna()].reset_index(
        drop=True)

    # modify the flowamounts related to the 6 naics 'orchards' are mapped to
    fba_w_sector = modify_orchard_flowamounts(fba_w_sector,
                                              activity_column=activity_col)
    return fba_w_sector


def modify_orchard_flowamounts(fba, activity_column):
    """
    In the CoA cropland crosswalk, the activity 'orchards' is mapped
    to eight 6-digit naics. Therefore, after mapping,
    divide the orchard flow amount by 8.
    :param fba: A FlowByActiivty df mapped to sectors
    :param activity_column: The activity column to base FlowAmount
        modifications on (ActivityProducedBy or ActivityConsumedBy)
    :return: df, CoA cropland data with modified FlowAmounts
    """

    # divide the Orchards data allocated to NAICS by 6 to avoid double counting
    fba['FlowAmount'] = np.where(fba[activity_column] == 'ORCHARDS',
                                 fba['FlowAmount'] / 8,
                                 fba['FlowAmount'])

    return fba
