# USDA_CoA_Cropland.py (flowsa)
# !/usr/bin/env python3
# coding=utf-8

"""
Functions used to import and parse USDA Census of Ag Cropland data
in NAICS format
"""

import json
import numpy as np
import pandas as pd

from flowsa.allocation import allocate_by_sector, \
    equally_allocate_parent_to_child_naics
from flowsa.common import WITHDRAWN_KEYWORD, US_FIPS, abbrev_us_state, \
    fba_wsec_default_grouping_fields, fbs_fill_na_dict, \
    fbs_default_grouping_fields
from flowsa.dataclean import replace_NoneType_with_empty_cells, \
    replace_strings_with_NoneType, clean_df
from flowsa.flowbyfunctions import assign_fips_location_system, \
    equally_allocate_suppressed_parent_to_child_naics, \
    load_fba_w_standardized_units, sector_aggregation, sector_disaggregation, \
    sector_ratios
from flowsa.schema import flow_by_sector_fields
from flowsa.sectormapping import add_sectors_to_flowbyactivity
from flowsa.validation import compare_df_units


def CoA_Cropland_NAICS_URL_helper(*, build_url, config, **_):
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
        # at national level, remove the text string calling for state acronyms
        if x == 'NATIONAL':
            url = build_url
            url = url.replace("__aggLevel__", x)
            url = url.replace("&state_alpha=__stateAlpha__", "")
            url = url.replace(" ", "%20")
            urls.append(url)
        else:
            # substitute in state acronyms for state and county url calls
            for z in state_abbrevs:
                url = build_url
                url = url.replace("__aggLevel__", x)
                url = url.replace("__stateAlpha__", z)
                url = url.replace(" ", "%20")
                urls.append(url)
    return urls


def coa_cropland_NAICS_call(*, resp, **_):
    """
    Convert response for calling url to pandas dataframe,
    begin parsing df into FBA format
    :param resp: df, response from url call
    :return: pandas dataframe of original source data
    """
    cropland_json = json.loads(resp.text)
    df_cropland = pd.DataFrame(data=cropland_json["data"])
    return df_cropland


def coa_cropland_NAICS_parse(*, df_list, year, **_):
    """
    Combine, parse, and format the provided dataframes
    :param df_list: list of dataframes to concat and format
    :param year: year
    :return: df, parsed and partially formatted to flowbyactivity
        specifications
    """
    df = pd.concat(df_list, sort=False)
    # specify desired data based on domain_desc
    df = df[df['domain_desc'] == 'NAICS CLASSIFICATION']
    # only want ag land and farm operations
    df = df[df['short_desc'].str.contains("AG LAND|FARM OPERATIONS")]
    # drop unused columns
    df = df.drop(columns=['agg_level_desc', 'location_desc', 'state_alpha',
                          'sector_desc', 'country_code', 'begin_code',
                          'watershed_code', 'reference_period_desc',
                          'asd_desc', 'county_name', 'source_desc',
                          'congr_district_code', 'asd_code', 'week_ending',
                          'freq_desc', 'load_time', 'zip_5',
                          'watershed_desc', 'region_desc', 'state_ansi',
                          'state_name', 'country_name', 'county_ansi',
                          'end_code', 'group_desc'])
    # create FIPS column by combining existing columns
    df.loc[df['county_code'] == '', 'county_code'] = '000'
    df['Location'] = df['state_fips_code'] + df['county_code']
    df.loc[df['Location'] == '99000', 'Location'] = US_FIPS
    # NAICS classification data
    # flowname
    df.loc[:, 'FlowName'] = df['commodity_desc'] + ', ' + \
                            df['class_desc'] + ', ' + df['prodn_practice_desc']
    df.loc[:, 'FlowName'] = df['FlowName'].str.replace(
        ", ALL PRODUCTION PRACTICES", "", regex=True)
    df.loc[:, 'FlowName'] = df['FlowName'].str.replace(
        ", ALL CLASSES", "", regex=True)
    # activity consumed/produced by
    df.loc[:, 'Activity'] = df['domaincat_desc']
    df.loc[:, 'Activity'] = df['Activity'].str.replace(
        "NAICS CLASSIFICATION: ", "", regex=True)
    df.loc[:, 'Activity'] = df['Activity'].str.replace('[()]+', '', regex=True)
    df['ActivityProducedBy'] = np.where(
        df["unit_desc"] == 'OPERATIONS', df["Activity"], '')
    df['ActivityConsumedBy'] = np.where(
        df["unit_desc"] == 'ACRES', df["Activity"], '')

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
    df['Spread'] = df['Spread'].str.strip()  # trim whitespace
    df.loc[df['Spread'] == "(H)", 'Spread'] = 99.95
    df.loc[df['Spread'] == "(L)", 'Spread'] = 0.05
    df.loc[df['Spread'] == "", 'Spread'] = None
    df.loc[df['Spread'] == "(D)", 'Spread'] = WITHDRAWN_KEYWORD
    # drop Descriptions that contain certain phrases, as these
    # data are included in other categories
    df = df[~df['Description'].str.contains(
        'FRESH MARKET|PROCESSING|ENTIRE CROP|NONE OF CROP|PART OF CROP')]
    # drop Descriptions that contain certain phrases -
    # only occur in AG LAND data
    df = df[~df['Description'].str.contains(
        'INSURANCE|OWNED|RENTED|FAILED|FALLOW|IDLE')].reset_index(drop=True)
    # add location system based on year of data
    df = assign_fips_location_system(df, year)
    # Add hardcoded data
    df['Class'] = np.where(df["Unit"] == 'ACRES', "Land", "Other")
    df['SourceName'] = "USDA_CoA_Cropland_NAICS"
    df['MeasureofSpread'] = "RSD"
    df['DataReliability'] = 5  # tmp
    df['DataCollection'] = 2
    return df


def coa_cropland_naics_fba_wsec_cleanup(fba_w_sector, **kwargs):
    """
    Clean up the land fba for use in allocation
    :param fba_w_sector: df, coa cropland naics flowbyactivity
        with sector columns
    :param kwargs: dictionary, requires df sourcename
    :return: df, flowbyactivity with modified values
    """

    df = equally_allocate_suppressed_parent_to_child_naics(
        fba_w_sector, 'SectorConsumedBy', fba_wsec_default_grouping_fields)
    return df


def disaggregate_df_to_naics6_w_cropland_naics(
        df_to_disag, coa_niacs_df, attr):
    """
    Disaggregate a df (typically usda coa cropland) to naics 6 using the
    cropland naics data
    :param fba_w_sector: df, CoA cropland data, FBA format with sector columns
    :param attr: dictionary, attribute data from method yaml for activity set
    :param method: dictionary, FBS method yaml
    :param kwargs: dictionary, arguments that might be required for other functions.
           Currently includes data source name.
    :return: df, CoA cropland with disaggregated NAICS sectors
    """
    # define the activity and sector columns to base modifications on
    # these definitions will vary dependent on class type
    sector_col = 'SectorConsumedBy'


    # use ratios of usda 'land in farms' to determine animal use of
    # pasturelands at 6 digit naics
    df = disaggregate_pastureland(df_to_disag, coa_niacs_df, attr,
                                  sector_column=sector_col,
                                  parameter_drop=['1125'])

    # use ratios of usda 'harvested cropland' to determine missing 6 digit
    # naics
    df2 = disaggregate_cropland(df, coa_niacs_df, sector_column=sector_col)

    return df2


def disaggregate_pastureland(df_to_disag, coa_naics, attr,
                             sector_column, **kwargs):
    """
    The USDA CoA Cropland irrigated pastureland data only links
    to the 3 digit NAICS '112'. This function uses state
    level CoA 'Land in Farms' to allocate the county level acreage data to
    6 digit NAICS.
    :param fba_w_sector: df, the CoA Cropland dataframe after linked to sectors
    :param attr: dictionary, attribute data from method yaml for activity set
    :param method: string, methodname
    :param year: str, year of data being disaggregated
    :param sector_column: str, the sector column on which to make df
                          modifications (SectorProducedBy or SectorConsumedBy)
    :param download_FBA_if_missing: bool, if True will attempt to load
        FBAS used in generating the FBS from remote server prior to
        generating if file not found locally
    :return: df, the CoA cropland dataframe with disaggregated pastureland data
    """

    # tmp drop NoneTypes
    df_to_disag = replace_NoneType_with_empty_cells(df_to_disag)

    # subset the coa data so only pastureland
    p = df_to_disag.loc[df_to_disag[sector_column].apply(
        lambda x: x[0:3]) == '112'].reset_index(drop=True)
    if len(p) != 0:
        # add temp loc column for state fips
        p = p.assign(Location_tmp=p['Location'].apply(lambda x: x[0:2]))

        # subset to land in farms data
        df_f = coa_naics[coa_naics['FlowName'] == 'FARM OPERATIONS']
        # subset to rows related to pastureland
        df_f = df_f.loc[df_f['ActivityConsumedBy'].apply(
            lambda x: x[0:3]) == '112']
        # drop rows with "&'
        df_f = df_f[~df_f['ActivityConsumedBy'].str.contains('&')]
        if 'parameter_drop' in kwargs:
            # drop aquaculture because pastureland not used for aquaculture
            df_f = df_f[~df_f['ActivityConsumedBy'].isin(kwargs['parameter_drop'])]
        # estimate suppressed data by equal allocation
        df_f = equally_allocate_suppressed_parent_to_child_naics(
            df_f, 'SectorConsumedBy', fba_wsec_default_grouping_fields)
        # create proportional ratios
        group_cols = [e for e in fba_wsec_default_grouping_fields if
                      e not in ('ActivityProducedBy', 'ActivityConsumedBy')]
        df_f = allocate_by_sector(df_f, attr, 'proportional', group_cols)
        # tmp drop NoneTypes
        df_f = replace_NoneType_with_empty_cells(df_f)
        # drop naics = '11
        df_f = df_f[df_f[sector_column] != '11']
        # drop 000 in location
        df_f = df_f.assign(Location=df_f['Location'].apply(lambda x: x[0:2]))

        # check units before merge
        compare_df_units(p, df_f)
        # merge the coa pastureland data with land in farm data
        df = p.merge(df_f[[sector_column, 'Location', 'FlowAmountRatio']],
                     how='left', left_on="Location_tmp", right_on="Location")
        # multiply the flowamount by the flowratio
        df.loc[:, 'FlowAmount'] = df['FlowAmount'] * df['FlowAmountRatio']
        # drop columns and rename
        df = df.drop(columns=['Location_tmp', sector_column + '_x',
                              'Location_y', 'FlowAmountRatio'])
        df = df.rename(columns={sector_column + '_y': sector_column,
                                "Location_x": 'Location'})

        # drop rows where sector = 112 and then concat with
        # original fba_w_sector
        fba_w_sector = df_to_disag[df_to_disag[sector_column].apply(
            lambda x: x[0:3]) != '112'].reset_index(drop=True)
        fba_w_sector = pd.concat([fba_w_sector, df],
                                 sort=True).reset_index(drop=True)

        # fill empty cells with NoneType
        fba_w_sector = replace_strings_with_NoneType(fba_w_sector)

    return fba_w_sector


def disaggregate_cropland(df_to_disag, coa_naics, sector_column):
    """
    In the event there are 4 (or 5) digit naics for cropland
    at the county level, use state level harvested cropland to
    create ratios
    :param fba_w_sector: df, CoA cropland data, FBA format with sector columns
    :param attr: dictionary, attribute data from method yaml for activity set
    :param method: string, method name
    :param year: str, year of data
    :param sector_column: str, the sector column on which to make
        df modifications (SectorProducedBy or SectorConsumedBy)
    :param download_FBA_if_missing: bool, if True will attempt to
        load FBAS used in generating the FBS from remote server prior to
        generating if file not found locally
    :return: df, CoA cropland data disaggregated
    """

    # tmp drop NoneTypes
    fba_w_sector = replace_NoneType_with_empty_cells(df_to_disag)

    # drop pastureland data
    crop = fba_w_sector.loc[fba_w_sector[sector_column].apply(
        lambda x: x[0:3]) != '112'].reset_index(drop=True)
    # drop sectors < 4 digits
    crop = crop[crop[sector_column].apply(
        lambda x: len(x) > 3)].reset_index(drop=True)
    # create tmp location
    crop = crop.assign(Location_tmp=crop['Location'].apply(lambda x: x[0:2]))

    # subset the harvested cropland by naics
    naics = coa_naics[
        coa_naics['FlowName'] == 'AG LAND, CROPLAND, HARVESTED'].reset_index(
        drop=True)
    # drop the activities that include '&'
    naics = naics[~naics['ActivityConsumedBy'].str.contains('&')].reset_index(drop=True)
    # estimate suppressed data by equally allocating parent to child naics
    naics = equally_allocate_suppressed_parent_to_child_naics(
        naics, 'SectorConsumedBy', fba_wsec_default_grouping_fields)
    # add missing fbs fields
    naics = clean_df(naics, flow_by_sector_fields, fbs_fill_na_dict)

    # aggregate sectors to create any missing naics levels
    group_cols = fbs_default_grouping_fields
    naics2 = sector_aggregation(naics)
    # add missing naics5/6 when only one naics5/6 associated with a naics4
    naics3 = sector_disaggregation(naics2)
    # drop rows where FlowAmount 0
    naics3 = naics3.loc[naics3['FlowAmount'] != 0]
    # create ratios
    naics4 = sector_ratios(naics3, sector_column)
    # create temporary sector column to match the two dfs on
    naics4 = naics4.assign(
        Location_tmp=naics4['Location'].apply(lambda x: x[0:2]))
    # tmp drop Nonetypes
    naics4 = replace_NoneType_with_empty_cells(naics4)

    # check units in prep for merge
    compare_df_units(crop, naics4)
    # for loop through naics lengths to determine
    # naics 4 and 5 digits to disaggregate
    for i in range(4, 6):
        # subset df to sectors with length = i and length = i + 1
        crop_subset = crop.loc[crop[sector_column].apply(
            lambda x: i + 1 >= len(x) >= i)]
        crop_subset = crop_subset.assign(
            Sector_tmp=crop_subset[sector_column].apply(lambda x: x[0:i]))
        # if duplicates drop all rows
        df = crop_subset.drop_duplicates(
            subset=['Location', 'Sector_tmp'],
            keep=False).reset_index(drop=True)
        # drop sector temp column
        df = df.drop(columns=["Sector_tmp"])
        # subset df to keep the sectors of length i
        df_subset = df.loc[df[sector_column].apply(lambda x: len(x) == i)]
        # subset the naics df where naics length is i + 1
        naics_subset = \
            naics4.loc[naics4[sector_column].apply(
                lambda x: len(x) == i + 1)].reset_index(drop=True)
        naics_subset = naics_subset.assign(
            Sector_tmp=naics_subset[sector_column].apply(lambda x: x[0:i]))
        # merge the two df based on locations
        df_subset = pd.merge(
            df_subset, naics_subset[[sector_column, 'FlowAmountRatio',
                                     'Sector_tmp', 'Location_tmp']],
            how='left', left_on=[sector_column, 'Location_tmp'],
            right_on=['Sector_tmp', 'Location_tmp'])
        # create flow amounts for the new NAICS based on the flow ratio
        df_subset.loc[:, 'FlowAmount'] = \
            df_subset['FlowAmount'] * df_subset['FlowAmountRatio']
        # drop rows of 0 and na
        df_subset = df_subset[df_subset['FlowAmount'] != 0]
        df_subset = df_subset[
            ~df_subset['FlowAmount'].isna()].reset_index(drop=True)
        # drop columns
        df_subset = df_subset.drop(
            columns=[sector_column + '_x', 'FlowAmountRatio', 'Sector_tmp'])
        # rename columns
        df_subset = df_subset.rename(
            columns={sector_column + '_y': sector_column})
        # tmp drop Nonetypes
        df_subset = replace_NoneType_with_empty_cells(df_subset)
        # add new rows of data to crop df
        crop = pd.concat([crop, df_subset], sort=True).reset_index(drop=True)

    # clean up df
    crop = crop.drop(columns=['Location_tmp'])

    # equally allocate any further missing naics
    crop = equally_allocate_parent_to_child_naics(crop, 'NAICS_6')

    # pasture data
    pasture = \
        fba_w_sector.loc[fba_w_sector[sector_column].apply(
            lambda x: x[0:3]) == '112'].reset_index(drop=True)
    # concat crop and pasture
    fba_w_sector = pd.concat([pasture, crop], sort=True).reset_index(drop=True)

    # fill empty cells with NoneType
    fba_w_sector = replace_strings_with_NoneType(fba_w_sector)

    return fba_w_sector
