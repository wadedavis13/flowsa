# fbs_allocation.py (flowsa)
# !/usr/bin/env python3
# coding=utf-8
"""
Functions to allocate data using additional data sources
"""

import numpy as np
import pandas as pd
from flowsa.common import US_FIPS, fba_activity_fields, \
    fbs_activity_fields, fba_mapped_wsec_default_grouping_fields, \
    fba_wsec_default_grouping_fields, check_activities_sector_like, \
    return_bea_codes_used_as_naics
from flowsa.schema import activity_fields, flow_by_activity_mapped_wsec_fields
from flowsa.settings import log
from flowsa.validation import check_allocation_ratios, \
    check_if_location_systems_match, compare_df_units
from flowsa.flowbyfunctions import collapse_activity_fields, \
    dynamically_import_fxn, sector_aggregation, sector_disaggregation, \
    subset_df_by_geoscale, return_primary_sector_column, \
    load_fba_w_standardized_units, aggregator
from flowsa.allocation import allocate_by_sector, \
    proportional_allocation_by_location_and_activity, \
    equally_allocate_parent_to_child_naics, equal_allocation
from flowsa.sectormapping import get_fba_allocation_subset, \
    add_sectors_to_flowbyactivity
from flowsa.dataclean import replace_strings_with_NoneType, add_missing_flow_by_fields
from flowsa.validation import check_if_data_exists_at_geoscale


def direct_allocation_method(fbs, k, names, method):
    """
    Directly assign activities to sectors
    :param fbs: df, FBA with flows converted using fedelemflowlist
    :param k: str, source name
    :param names: list, activity names in activity set
    :param method: dictionary, FBS method yaml
    :return: df with sector columns
    """
    log.info('Directly assigning activities to sectors')
    # for each activity, if activities are not sector like,
    # check that there is no data loss
    if check_activities_sector_like(k) is False:
        activity_list = []
        n_allocated = []
        for n in names:
            # avoid double counting by dropping n from the df after calling on
            # n, in the event both ACB and APB values exist
            fbs = fbs[~((fbs[fba_activity_fields[0]].isin(n_allocated)) |
                      (fbs[fba_activity_fields[1]].isin(n_allocated))
                        )].reset_index(drop=True)
            log.debug('Checking for %s at %s',
                      n, method['target_sector_level'])
            fbs_subset = \
                fbs[(fbs[fba_activity_fields[0]] == n) |
                    (fbs[fba_activity_fields[1]] == n)].reset_index(drop=True)
            # check if an Activity maps to more than one sector,
            # if so, equally allocate
            fbs_subset = equal_allocation(fbs_subset)
            fbs_subset = equally_allocate_parent_to_child_naics(fbs_subset, method['target_sector_level'])
            activity_list.append(fbs_subset)
            n_allocated.append(n)
        fbs = pd.concat(activity_list, ignore_index=True)
    return fbs


def function_allocation_method(flow_subset_mapped, primary_source, names,
                               attr, fbs_list):
    """
    Allocate df activities to sectors using a function identified
    in the FBS method yaml
    :param flow_subset_mapped: df, FBA with flows converted using
        fedelemflowlist
    :param primary_source: str, source name
    :param names: list, activity names in activity set
    :param attr: dictionary, attribute data from method yaml for activity set
    :param fbs_list: list, fbs dfs created running flowbysector.py
    :return: df, FBS, with allocated activity columns to sectors
    """
    log.info('Calling on function specified in method yaml to allocate '
             '%s to sectors', ', '.join(map(str, names)))
    fbs = dynamically_import_fxn(primary_source, attr['allocation_source'])(
        flow_subset_mapped, attr, fbs_list)
    return fbs


def load_clean_allocation_fba(df_to_modify, alloc_method, alloc_config,
                              names, method, primary_source, primary_config,
                              download_FBA_if_missing):
    """

    :param df_to_modify:
    :param alloc_method:
    :param alloc_config:
    :param names:
    :param method:
    :param primary_source:
    :param primary_config:
    :param download_FBA_if_missing:
    :return:
    """
    # add parameters to dictionary if exist in method yaml
    fba_dict = {}
    if 'flow' in alloc_config:
        fba_dict['flow'] = alloc_config['flow']
    if 'compartment' in alloc_config:
        fba_dict['compartment'] = alloc_config['compartment']
    if 'clean_fba_fxn' in alloc_config:
        fba_dict['clean_fba_fxn'] = alloc_config['clean_fba_fxn']
    if 'clean_fba_w_sec_fxn' in alloc_config:
        fba_dict['clean_fba_w_sec_fxn'] = alloc_config['clean_fba_w_sec_fxn']

    # load the allocation FBA
    fba_allocation_wsec = \
        load_map_clean_fba(method, alloc_config,
                           fba_sourcename=alloc_config['allocation_source'],
                           df_year=alloc_config['year'],
                           flowclass=alloc_config['class'],
                           geoscale_from=alloc_config['geographic_scale'],
                           geoscale_to=primary_config['geographic_scale'],
                           download_FBA_if_missing=download_FBA_if_missing,
                           **fba_dict)
    # run sector disagg to capture any missing lower level naics that have a
    # singular parent to child relationship
    fba_allocation_wsec = sector_disaggregation(fba_allocation_wsec)

    # subset fba datasets to only keep the sectors associated
    # with activity subset
    log.info("Subsetting %s for sectors in %s",
             alloc_config['allocation_source'], primary_source)
    fba_allocation_wsec_sub = get_fba_allocation_subset(
        fba_allocation_wsec, primary_source, names,
        flowSubsetMapped=df_to_modify, allocMethod=alloc_method)

    return fba_allocation_wsec_sub


def merge_fbas_by_geoscale(df1, df1_geoscale, df2, df2_geoscale):


    # generalize activity field names to enable link to main fba source
    log.info('Generalizing activity columns')
    df2 = collapse_activity_fields(df2)
    # rename column
    df2 = df2.rename(columns={"FlowAmount": 'HelperFlow'})

    sector_col_to_merge = return_primary_sector_column(df1)
    # check df units
    compare_df_units(df1, df2)
    # merge allocation df with helper df based on sectors,
    # depending on geo scales of dfs
    if (df2_geoscale == 'state') and (df1_geoscale == 'county'):
        df2['Location_tmp'] = df2['Location'].apply(lambda x: x[0:2])
        df1['Location_tmp'] = df1['Location'].apply(lambda x: x[0:2])
        dfm = df1.merge(df2[['Location_tmp', 'Sector', 'HelperFlow']],
                        how='left',
                        left_on=['Location_tmp', sector_col_to_merge],
                        right_on=['Location_tmp', 'Sector'])
        dfm = dfm.drop(columns=['Location_tmp'])
    elif (df2_geoscale == 'national') and (df1_geoscale != 'national'):
        dfm = df1.merge(df2[['Sector', 'HelperFlow']], how='left',
                        left_on=[sector_col_to_merge], right_on=['Sector'])
    else:
        dfm = df1.merge(df2[['Location', 'Sector', 'HelperFlow']],
                        left_on=['Location', sector_col_to_merge],
                        right_on=['Location', 'Sector'],
                        how='left')
    # load bea codes that sub for naics
    bea = return_bea_codes_used_as_naics()
    # replace sector column and helperflow value if the sector column to
    # merge is in the bea list to prevent dropped data
    # todo: check if next step works
    dfm['HelperFlow'] = np.where(dfm[sector_col_to_merge].isin(bea),
                                 dfm['FlowAmount'],
                                 dfm['HelperFlow'])
    dfm = dfm.drop(columns=['Sector'])
    # drop all rows where helperflow is null
    dfm2 = dfm.dropna(subset=['HelperFlow']).reset_index(drop=True)

    return dfm2


def dataset_allocation_method(flow_subset_mapped, attr, names, method,
                              primary_source, primary_config, aset,
                              aset_names, download_FBA_if_missing):
    """
    Method of allocation using a specified data source
    :param flow_subset_mapped: FBA subset mapped using federal
        elementary flow list
    :param attr: dictionary, attribute data from method yaml for activity set
    :param names: list, activity names in activity set
    :param method: dictionary, FBS method yaml
    :param primary_source: str, the datasource name
    :param primary_config: dictionary, the datasource parameters
    :param aset: dictionary items for FBS method yaml
    :param aset_names: list, activity set names
    :param download_FBA_if_missing: bool, indicate if missing FBAs
       should be downloaded from Data Commons
    :return: df, allocated activity names
    """

    # define a list of allocation methods where the two allocation FBAs
    # should not be merged/should not determine a "helper flow" prior to the
    # allocation method
    allocation_exception_list = ['disaggregation', 'weighted_avg']
    # todo: modify so can have unlimited number of allocation methods..
    # determine the allocation methods used to modify the source data. Loop
    # through the methods and loop through any further allocation methods
    # before modifying the source dataset
    for alloc_method1, alloc_config1 in attr['allocation_method'].items():
        alloc_df1 = load_clean_allocation_fba(
            flow_subset_mapped, alloc_method1, alloc_config1, names, method,
            primary_source, primary_config, download_FBA_if_missing)
        if 'allocation_method' in alloc_config1:
            for alloc_method2, alloc_config2 in \
                    alloc_config1['allocation_method'].items():
                alloc_df2 = load_clean_allocation_fba(
                    alloc_df1, alloc_method2, alloc_config2, names, method,
                    primary_source, primary_config, download_FBA_if_missing)
                if 'allocation_method' in alloc_config2:
                    for alloc_method3, alloc_config3 in \
                            alloc_config2['allocation_method'].items():
                        alloc_df3 = load_clean_allocation_fba(
                            alloc_df2, alloc_method3, alloc_config3, names,
                            method, primary_source, primary_config,
                            download_FBA_if_missing)
                        if alloc_method3 not in allocation_exception_list:
                            alloc_df2 = merge_fbas_by_geoscale(
                                alloc_df2, alloc_config2['geographic_scale'],
                                alloc_df3, alloc_config3['geographic_scale'])
                        alloc_df2 = allocate_source_w_secondary_source(
                            alloc_df2, alloc_df3, alloc_method3, alloc_config3)
                if alloc_method2 not in allocation_exception_list:
                    alloc_df1 = merge_fbas_by_geoscale(
                        alloc_df1, alloc_config1['geographic_scale'],
                        alloc_df2, alloc_config2['geographic_scale'])
                alloc_df1 = allocate_source_w_secondary_source(
                    alloc_df1, alloc_df2, alloc_method2, alloc_config2)
        if alloc_method1 not in allocation_exception_list:
            if 'geographic_scale' in attr:
                activity_geoscale = attr.get('geographic_scale')
            else:
                activity_geoscale = primary_config['geographic_scale']
            flow_subset_mapped = merge_fbas_by_geoscale(
                flow_subset_mapped, activity_geoscale,
                alloc_df1, alloc_config1['geographic_scale'])
        flow_subset_mapped = allocate_source_w_secondary_source(
            flow_subset_mapped, alloc_df1, alloc_method1, alloc_config1)

    return flow_subset_mapped


def allocate_source_w_secondary_source(primary_df, secondary_df,
                                       allocation_method,
                                       allocation_configuration):

    # determine sector column with values
    sector_col = return_primary_sector_column(primary_df)
    # modify flow amounts using helper data
    if allocation_method == 'multiplication':
        df = fba_multiplication(primary_df, sector_col)
    if allocation_method == 'proportional':
        df = fba_proportional(primary_df, sector_col)
    if allocation_method == 'proportional-flagged':
        df = fba_proportional_flagged(primary_df)
    if allocation_method == 'disaggregation':
        df = fba_disaggregation(primary_df, secondary_df,
                                allocation_configuration)
    if allocation_method == 'weighted_avg':
        df = fba_weighted_avg(primary_df, secondary_df,
                              allocation_configuration)
    # option to scale up fba values
    if allocation_method == 'scaled':
        df = fba_scale(primary_df)

    # reset df to only have standard columns
    df2 = add_missing_flow_by_fields(df, flow_by_activity_mapped_wsec_fields)
    # aggregate df
    df3 = aggregator(df2, fba_mapped_wsec_default_grouping_fields)

    return df3


def fba_multiplication(df_load, sector_col):
    # if missing values (na or 0), replace with national level values
    replacement_values = df_load[df_load['Location'] == US_FIPS]
    replacement_values = replacement_values.rename(
        columns={"HelperFlow": 'ReplacementValue'})
    modified_fba_allocation = df_load.merge(
        replacement_values[[sector_col, 'ReplacementValue']], how='left')
    modified_fba_allocation.loc[:, 'HelperFlow'] = \
        modified_fba_allocation['HelperFlow'].fillna(
        modified_fba_allocation['ReplacementValue'])
    modified_fba_allocation.loc[:, 'HelperFlow'] =\
        np.where(modified_fba_allocation['HelperFlow'] == 0,
                 modified_fba_allocation['ReplacementValue'],
                 modified_fba_allocation['HelperFlow'])

    # replace non-existent helper flow values with a 0,
    # so after multiplying, don't have incorrect value associated with
    # new unit
    modified_fba_allocation['HelperFlow'] =\
        modified_fba_allocation['HelperFlow'].fillna(value=0)
    modified_fba_allocation.loc[:, 'FlowAmount'] = \
        modified_fba_allocation['FlowAmount'] * \
        modified_fba_allocation['HelperFlow']
    return modified_fba_allocation


def fba_proportional(df_load, col_for_alloc_ratios):
    modified_fba_allocation = proportional_allocation_by_location_and_activity(
            df_load, col_for_alloc_ratios)
    modified_fba_allocation.loc[:, 'FlowAmount'] = \
        modified_fba_allocation['FlowAmount'] * \
        modified_fba_allocation['FlowAmountRatio']
    return modified_fba_allocation


def fba_proportional_flagged(df_load):
    # calculate denominators based on activity and 'flagged' column
    modified_fba_allocation =df_load.assign(
            Denominator=df_load.groupby(
                ['FlowName', 'ActivityConsumedBy', 'Location',
                 'disaggregate_flag'])['HelperFlow'].transform('sum'))
    modified_fba_allocation = modified_fba_allocation.assign(
        FlowAmountRatio=modified_fba_allocation['HelperFlow'] /
                        modified_fba_allocation['Denominator'])
    modified_fba_allocation =\
        modified_fba_allocation.assign(
            FlowAmount=modified_fba_allocation['FlowAmount'] *
                       modified_fba_allocation['FlowAmountRatio'])
    modified_fba_allocation =\
        modified_fba_allocation.drop(
            columns=['disaggregate_flag', 'Sector', 'HelperFlow',
                     'Denominator', 'FlowAmountRatio'])
    # run sector aggregation
    modified_fba_allocation = \
        sector_aggregation(modified_fba_allocation)

    return modified_fba_allocation


def fba_disaggregation(primary_df, secondary_df, allocation_configuration):
    df = dynamically_import_fxn(
        allocation_configuration['allocation_source'],
        allocation_configuration['disaggregation_fxn']
    )(primary_df, secondary_df, allocation_configuration)

    return df


def fba_weighted_avg(primary_df, secondary_df, allocation_configuration):
    df = dynamically_import_fxn(
        allocation_configuration['allocation_source'],
        allocation_configuration['weighted_avg_fxn']
    )(primary_df, secondary_df)
    return df


def fba_scale(df_load):
    log.info("Scaling %s to FBA values")
    modified_fba_allocation = \
        dynamically_import_fxn(
            attr['allocation_source'], attr["scale_helper_results"])(
            df_load, attr,
            download_FBA_if_missing=download_FBA_if_missing)
    return modified_fba_allocation


def load_map_clean_fba(method, attr, fba_sourcename, df_year, flowclass,
                       geoscale_from, geoscale_to, **kwargs):
    """
    Load, clean, and map a FlowByActivity df
    :param method: dictionary, FBS method yaml
    :param attr: dictionary, attribute data from method yaml for activity set
    :param fba_sourcename: str, source name
    :param df_year: str, year
    :param flowclass: str, flowclass to subset df with
    :param geoscale_from: str, geoscale to use
    :param geoscale_to: str, geoscale to aggregate to
    :param kwargs: dictionary, can include parameters: 'allocation_flow',
                   'allocation_compartment','clean_allocation_fba',
                   'clean_allocation_fba_w_sec'
    :return: df, fba format
    """
    # dictionary to load/standardize fba
    kwargs_dict = {}
    if 'download_FBA_if_missing' in kwargs:
        kwargs_dict['download_FBA_if_missing'] = \
            kwargs['download_FBA_if_missing']
    if 'allocation_map_to_flow_list' in attr:
        kwargs_dict['allocation_map_to_flow_list'] = \
            attr['allocation_map_to_flow_list']

    log.info("Loading allocation flowbyactivity %s for year %s",
             fba_sourcename, str(df_year))
    fba = load_fba_w_standardized_units(datasource=fba_sourcename,
                                        year=df_year,
                                        flowclass=flowclass,
                                        **kwargs_dict
                                        )

    # check if allocation data exists at specified geoscale to use
    log.info("Checking if allocation data exists at the %s level",
             geoscale_from)
    check_if_data_exists_at_geoscale(fba, geoscale_from)

    # aggregate geographically to the scale of the flowbyactivty source,
    # if necessary
    fba = subset_df_by_geoscale(fba, geoscale_from, geoscale_to)

    # subset based on yaml settings
    if 'flow' in kwargs:
        if kwargs['flow'] != 'None':
            fba = fba.loc[fba['FlowName'].isin(kwargs['flow'])]
    if 'compartment' in kwargs:
        if kwargs['compartment'] != 'None':
            fba = fba.loc[fba['Compartment'].isin(kwargs['compartment'])]

    # cleanup the fba allocation df, if necessary
    if 'clean_fba_fxn' in kwargs:
        log.info("Cleaning %s", fba_sourcename)
        fba = dynamically_import_fxn(fba_sourcename, kwargs["clean_fba_fxn"])(
            fba, attr=attr,
            download_FBA_if_missing=kwargs['download_FBA_if_missing'])
    # reset index
    fba = fba.reset_index(drop=True)

    # assign sector to allocation dataset
    log.info("Adding sectors to %s", fba_sourcename)
    fba_wsec = add_sectors_to_flowbyactivity(fba, sectorsourcename=method[
        'target_sector_source'])

    # call on fxn to further clean up/disaggregate the fba
    # allocation data, if exists
    if 'clean_fba_w_sec_fxn' in kwargs:
        log.info("Further disaggregating sectors in %s", fba_sourcename)
        fba_wsec = dynamically_import_fxn(
            fba_sourcename, kwargs['clean_fba_w_sec_fxn'])(
            fba_wsec, attr=attr, method=method, sourcename=fba_sourcename,
            download_FBA_if_missing=kwargs['download_FBA_if_missing'])

    return fba_wsec
