# fbs_allocation.py (flowsa)
# !/usr/bin/env python3
# coding=utf-8
"""
Functions to allocate data using additional data sources
"""

import numpy as np
import pandas as pd
import re
from flowsa.allocation import equal_allocation, \
    proportional_allocation_by_location_and_activity, \
    equally_allocate_parent_to_child_naics
from flowsa.common import fba_mapped_wsec_default_grouping_fields, \
    check_activities_sector_like, return_bea_codes_used_as_naics, \
    load_crosswalk, fbs_activity_fields
from flowsa.dataclean import add_missing_flow_by_fields, \
    replace_NoneType_with_empty_cells
from flowsa.flowbyfunctions import collapse_activity_fields, \
    dynamically_import_fxn, sector_aggregation, sector_disaggregation, \
    subset_df_by_geoscale, return_primary_sector_column, \
    load_fba_w_standardized_units, aggregator, \
    subset_df_by_sector_lengths, subset_and_merge_df_by_sector_lengths
from flowsa.location import US_FIPS
from flowsa.schema import flow_by_activity_mapped_wsec_fields
from flowsa.sectormapping import get_fba_allocation_subset, \
    add_sectors_to_flowbyactivity
from flowsa.settings import log
from flowsa.validation import compare_df_units, \
    check_for_data_loss_on_df_merge, check_if_data_exists_at_geoscale


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
        # check if an Activity maps to more than one sector,
        # if so, equally allocate
        fbs2 = equal_allocation(fbs)
        fbs3 = equally_allocate_parent_to_child_naics(
            fbs2, method)
    return fbs3


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


def load_allocation_fba(alloc_config, method, primary_config,
                        download_FBA_if_missing, subset_by_geoscale=True):
    """
    Method of allocation using a specified data source
    :param flow_subset_mapped: FBA subset mapped using federal
        elementary flow list
    :param attr: dictionary, attribute data from method yaml for activity set
    :param names: list, activity names in activity set
    :param method: dictionary, FBS method yaml
    :param k: str, the datasource name
    :param v: dictionary, the datasource parameters
    :param aset: dictionary items for FBS method yaml
    :param aset_names: list, activity set names
    :param download_FBA_if_missing: bool, indicate if missing FBAs
       should be downloaded from Data Commons
    :return: df, allocated activity names
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
                           subset_by_geoscale=subset_by_geoscale,
                           download_FBA_if_missing=download_FBA_if_missing,
                           fbsconfigpath=fbsconfigpath,
                           **fba_dict)
    return fba_allocation_wsec


def load_clean_allocation_fba(df_to_modify, alloc_method, alloc_config,
                              names, method, primary_source, primary_config,
                              download_FBA_if_missing,
                              subset_by_geoscale=True):
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
    fba_allocation_wsec = \
        load_allocation_fba(alloc_config, method, primary_config,
                        download_FBA_if_missing, subset_by_geoscale)
    # run sector disagg to capture any missing lower level naics that have a
    # singular parent to child relationship
    fba_allocation_wsec2 = sector_disaggregation(fba_allocation_wsec)

    # subset fba datasets to only keep the sectors associated
    # with activity subset
    log.info("Subsetting %s for sectors in %s",
             alloc_config['allocation_source'], primary_source)
    fba_allocation_wsec_sub = get_fba_allocation_subset(
        fba_allocation_wsec2, primary_source, names,
        sectorconfig=primary_config, flowSubsetMapped=df_to_modify,
        allocMethod=alloc_method)
    fba_allocation_wsec_sub2 = sector_disaggregation(fba_allocation_wsec_sub)

    # if the method calls for certain parameters in the allocation df to be
    # dropped, drop them here
    if 'drop_sectors' in alloc_config:
        # determine sector column with values
        sector_col = return_primary_sector_column(fba_allocation_wsec2)
        log.info('Dropping all %s that begin with %s from %s, used for %s '
                 'allocation of %s',
                 sector_col, alloc_config['drop_sectors'], alloc_config[
                     'allocation_source'], alloc_method, primary_source)
        fba_allocation_wsec_sub2 = fba_allocation_wsec_sub2[
            ~fba_allocation_wsec_sub2[sector_col].str.startswith(tuple(
                alloc_config['drop_sectors']))]

    return fba_allocation_wsec_sub2


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
    # note: do not drop rows where HelperFlow is 0 here because that 0 flow
    # is needed in multiplication allocation where the 0 is replaced with
    # data from other geoscales

    # determine if losing data during merge due to lack of data in the
    # secondary set
    check_for_data_loss_on_df_merge(df1, dfm)

    return dfm


def dataset_allocation_method(flow_subset_mapped, attr, names, method,
                              primary_source, primary_config, aset,
                              aset_names, download_FBA_if_missing,
                              fbsconfigpath):
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
    allocation_exception_list = ['proportional_disaggregation', 'weighted_avg']
    subset_by_geoscale = True
    # todo: modify so can have unlimited number of allocation methods..
    # determine the allocation methods used to modify the source data. Loop
    # through the methods and loop through any further allocation methods
    # before modifying the source dataset
    for alloc_method1, alloc_config1 in attr['allocation_method'].items():
        alloc_df1 = load_clean_allocation_fba(
            flow_subset_mapped, alloc_method1, alloc_config1, names, method,
            primary_source, primary_config, download_FBA_if_missing,
            subset_by_geoscale=subset_by_geoscale)
        if 'allocation_method' in alloc_config1:
            for alloc_method2, alloc_config2 in \
                    alloc_config1['allocation_method'].items():
                if alloc_method2 == 'multiplication':
                    subset_by_geoscale = False
                alloc_df2 = load_clean_allocation_fba(
                    alloc_df1, alloc_method2, alloc_config2, names, method,
                    primary_source, primary_config, download_FBA_if_missing,
                    subset_by_geoscale=subset_by_geoscale)
                if 'allocation_method' in alloc_config2:
                    for alloc_method3, alloc_config3 in \
                            alloc_config2['allocation_method'].items():
                        if alloc_method3 == 'multiplication':
                            subset_by_geoscale = False
                        alloc_df3 = load_clean_allocation_fba(
                            alloc_df2, alloc_method3, alloc_config3, names,
                            method, primary_source, primary_config,
                            download_FBA_if_missing,
                            subset_by_geoscale=subset_by_geoscale)
                        if alloc_method3 not in allocation_exception_list:
                            alloc_df2 = merge_fbas_by_geoscale(
                                alloc_df2, alloc_config2['geographic_scale'],
                                alloc_df3, alloc_config3['geographic_scale'])
                        alloc_df2 = allocate_source_w_secondary_source(
                            alloc_df2, alloc_config2, alloc_df3,
                            alloc_config3, alloc_method3, method)
                if alloc_method2 not in allocation_exception_list:
                    alloc_df1 = merge_fbas_by_geoscale(
                        alloc_df1, alloc_config1['geographic_scale'],
                        alloc_df2, alloc_config2['geographic_scale'])
                alloc_df1 = allocate_source_w_secondary_source(
                    alloc_df1, alloc_config1, alloc_df2, alloc_config2,
                    alloc_method2, method)
        if alloc_method1 not in allocation_exception_list:
            if 'geographic_scale' in attr:
                activity_geoscale = attr.get('geographic_scale')
            else:
                activity_geoscale = primary_config['geographic_scale']
            flow_subset_mapped = merge_fbas_by_geoscale(
                flow_subset_mapped, activity_geoscale,
                alloc_df1, alloc_config1['geographic_scale'])
        flow_subset_mapped = allocate_source_w_secondary_source(
            flow_subset_mapped, attr, alloc_df1, alloc_config1,
            alloc_method1, method)

    return flow_subset_mapped


def allocate_source_w_secondary_source(primary_df, primary_config,
                                       secondary_df, secondary_config,
                                       allocation_method, method):

    # first strip any "_XX" that might exist at the end of the method if a
    # dataset is modified using the same method for different subsets of
    # data. For example, the FBS method file might contain instructions for
    # "proportional_allocation_1" and "proportional_allocation_2"
    allocation_method = re.sub(r'_\d+$', '', allocation_method)

    # determine sector column with values
    sector_col = return_primary_sector_column(primary_df)
    # modify flow amounts using helper data
    if allocation_method == 'multiplication':
        df = fba_multiplication(primary_df, primary_config, secondary_df,
                                secondary_config, allocation_method,
                                sector_col, method)
    if allocation_method == 'proportional':
        df = fba_proportional(primary_df, primary_config, secondary_config,
                              sector_col, method)
    if allocation_method == 'proportional-flagged':
        df = fba_proportional_flagged(primary_df)
    if allocation_method == 'proportional_disaggregation':
        df = fba_proportional_disaggregation(primary_df, primary_config,
                                             secondary_df, secondary_config,
                                             sector_col, method)
    if allocation_method == 'weighted_avg':
        df = fba_weighted_avg(primary_df, secondary_df,
                              secondary_config)
    # option to scale up fba values
    if allocation_method == 'scaled':
        df = fba_scale(primary_df)

    # reset df to only have standard columns
    df2 = add_missing_flow_by_fields(df, flow_by_activity_mapped_wsec_fields)
    # aggregate df
    df3 = aggregator(df2, fba_mapped_wsec_default_grouping_fields)

    return df3


def fba_multiplication(primary_df, primary_config, secondary_df,
                       secondary_config, allocation_method, sector_col,
                       method):
    # todo: change to loop through geo scales one level at a time
    # if missing values (na or 0), replace with national level values
    # reload the secondary data source, this time skipping the geoscale subset
    # replacement_df = load_allocation_fba(
    #     secondary_config, method, primary_config,
    #     download_FBA_if_missing=False, subset_by_geoscale=False)
    replacement_df = secondary_df[secondary_df['Location'] ==
                                        US_FIPS].reset_index(drop=True)

    replacement_df = replacement_df.rename(
        columns={"FlowAmount": 'ReplacementValue'})
    modified_fba_allocation = primary_df.merge(
        replacement_df[[sector_col, 'ReplacementValue']], how='left')
    modified_fba_allocation.loc[:, 'HelperFlow'] = \
        modified_fba_allocation['HelperFlow'].fillna(
        modified_fba_allocation['ReplacementValue'])
    modified_fba_allocation.loc[:, 'HelperFlow'] =\
        np.where(modified_fba_allocation['HelperFlow'] == 0,
                 modified_fba_allocation['ReplacementValue'],
                 modified_fba_allocation['HelperFlow'])
    modified_fba_allocation = modified_fba_allocation.drop(
        columns='ReplacementValue')

    # todo: modify units
    # replace non-existent helper flow values with a 0,
    # so after multiplying, don't have incorrect value associated with
    # new unit
    modified_fba_allocation['HelperFlow'] =\
        modified_fba_allocation['HelperFlow'].fillna(value=0)
    modified_fba_allocation.loc[:, 'FlowAmount'] = \
        modified_fba_allocation['FlowAmount'] * \
        modified_fba_allocation['HelperFlow']
    # drop rows with flow = 0
    modified_fba_allocation2 = modified_fba_allocation[
        modified_fba_allocation['FlowAmount'] != 0].reset_index(drop=True)

    # determine if losing data during merge due to lack of data in the
    # secondary set
    check_for_data_loss_on_df_merge(primary_df, modified_fba_allocation2)

    return modified_fba_allocation2


def fba_proportional(primary_df, primary_config, secondary_config,
                     col_for_alloc_ratios, method):

    # drop all rows where helperflow is null
    df = primary_df.dropna(subset=['HelperFlow']).reset_index(drop=True)
    fba_mod = proportional_allocation_by_location_and_activity(
            df, method, primary_config, secondary_config, col_for_alloc_ratios)
    fba_mod['FlowAmount'] = fba_mod['FlowAmount'] * fba_mod['FlowAmountRatio']

    return fba_mod


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


def fba_proportional_disaggregation(primary_df, primary_config, secondary_df,
                                    secondary_config, col_for_alloc_ratios,
                                    method):
    # if the secondary configuration has a parameter indicating that the
    # secondary datasource is only meant to proportionally allocate a subset
    # of the primary df, then subset the primary df into the df sectors
    # meant to be allocated and another df not meant to be allocated,
    # concat after proportional allocation
    if 'df_subset_keep' in secondary_config:
        df = primary_df[primary_df[col_for_alloc_ratios].isin(
            secondary_config['df_subset_keep'])].reset_index(drop=True)
        df_prim = primary_df[~primary_df[col_for_alloc_ratios].isin(
            secondary_config['df_subset_keep'])].reset_index(drop=True)
    elif 'df_subset_drop' in secondary_config:
        df = primary_df[~primary_df[col_for_alloc_ratios].isin(
            secondary_config['df_subset_drop'])].reset_index(drop=True)
        df_prim = primary_df[primary_df[col_for_alloc_ratios].isin(
            secondary_config['df_subset_drop'])].reset_index(drop=True)
    else:
        df = primary_df.copy(deep=True)
        df_prim = pd.DataFrame()

    # determine the rows within the df to further disaggregate that require
    # disaggregation
    # load naics length crosswwalk
    # cw_load = load_crosswalk('sector_length')
    cw_load = load_crosswalk('sector_length')

    # find the longest length sector
    maxlength = df[[fbs_activity_fields[0], fbs_activity_fields[1]]].apply(
        lambda x: x.str.len()).max().max()
    maxlength = int(maxlength)
    if maxlength < 6:
        maxlength = maxlength + 1
    dfs_list = []
    for i in range(2, maxlength):
        sectors = cw_load[[f'NAICS_{str(i)}', f'NAICS_{str(i+1)}']].\
            drop_duplicates().reset_index(drop=True)

        # subset df by length of i and create temporary sector columns
        dfs1 = subset_df_by_sector_lengths(df, [i]).reset_index(drop=True)
        for s in ['Produced', 'Consumed']:
            dfs1 = dfs1.merge(sectors, how='left', left_on=[f'Sector{s}By'],
                              right_on=f'NAICS_{str(i)}').drop(
                columns=f'NAICS_{str(i)}')
            dfs1 = dfs1.rename(
                columns={f'NAICS_{str(i+1)}': f'Sector{s}By_tmp'})
        # drop any rows where there isn't a sector at a greater length
        dfs1 = dfs1.dropna(
            subset=['SectorProducedBy_tmp', 'SectorConsumedBy_tmp'],
            how='all')

        # subset df by length of i+1 and create temporary sector columns
        dfs2 = subset_df_by_sector_lengths(df, [i+1]).reset_index(drop=True)
        for s in ['Produced', 'Consumed']:
            dfs2 = dfs2.merge(
                sectors, how='left', left_on=[f'Sector{s}By'],
                right_on=f'NAICS_{str(i+1)}').drop(columns=f'NAICS_{str(i+1)}')
            dfs2 = dfs2.rename(columns={f'NAICS_{str(i)}': f'Sector{s}By_tmp'})
        dfs2 = replace_NoneType_with_empty_cells(dfs2)

        dfc = pd.concat([dfs1, dfs2], ignore_index=True)
        # if duplicates drop all rows
        dfc2 = dfc.drop_duplicates(subset=['Location',
                                           'SectorProducedBy_tmp',
                                           'SectorConsumedBy_tmp'],
                                   keep=False).reset_index(drop=True)
        # drop sector temp column
        dfc2 = dfc2.drop(columns=['SectorProducedBy_tmp',
                                  'SectorConsumedBy_tmp'])
        # subset df to keep the sectors of length i
        dfs3 = subset_df_by_sector_lengths(dfc2, [i]).reset_index(drop=True)
        # append to df
        dfs_list.append(dfs3)
    dfs = pd.concat(dfs_list, ignore_index=True)

    # drop rows in dfs from rows in df to avoid double counting - will
    # concat at the end
    df_nodisag = pd.merge(df, dfs, how='left', indicator=True).query(
        '_merge=="left_only"').drop('_merge', axis=1)

    # drop the sector columns in df subset that needs further disaggregation
    df1 = dfs.drop(columns=['SectorProducedBy', 'ProducedBySectorType',
                            'SectorConsumedBy', 'ConsumedBySectorType',
                            'SectorSourceName', 'HelperFlow'])
    # remap to sectors, this time assume activity is aggregated
    df2 = add_sectors_to_flowbyactivity(
        df1, sectorsourcename=method['target_sector_source'],
        overwrite_sectorlevel='aggregated')
    # merge with allocation df
    dfm = merge_fbas_by_geoscale(
        df2, primary_config['geographic_scale'],
        secondary_df, secondary_config['geographic_scale'])

    # drop all rows where helperflow is null
    dfm2 = dfm.dropna(subset=['HelperFlow']).reset_index(drop=True)
    fba_mod = proportional_allocation_by_location_and_activity(
        dfm2, method, primary_config, secondary_config, col_for_alloc_ratios)
    fba_mod['FlowAmount'] = fba_mod['FlowAmount'] * fba_mod['FlowAmountRatio']

    # if the df required subset, concat the two dfs
    if not df_prim.empty:
        fba_mod = pd.concat([fba_mod, df_nodisag, df_prim])

    return fba_mod


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
                       geoscale_from, geoscale_to, subset_by_geoscale=True,
                       fbsconfigpath=None, **kwargs):
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
    # if necessary unless fxn told to skip (default is true)
    if subset_by_geoscale:
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
    activity_to_sector_mapping = attr.get('activity_to_sector_mapping')
    if 'activity_to_sector_mapping' in kwargs:
        activity_to_sector_mapping = kwargs.get('activity_to_sector_mapping')
    log.info("Adding sectors to %s", fba_sourcename)
    fba_wsec = add_sectors_to_flowbyactivity(fba, sectorsourcename=method[
        'target_sector_source'],
        activity_to_sector_mapping=activity_to_sector_mapping,
        fbsconfigpath=fbsconfigpath)

    # call on fxn to further clean up/disaggregate the fba
    # allocation data, if exists
    if 'clean_fba_w_sec_fxn' in kwargs:
        log.info("Further disaggregating sectors in %s", fba_sourcename)
        fba_wsec = dynamically_import_fxn(
            fba_sourcename, kwargs['clean_fba_w_sec_fxn'])(
            fba_wsec, attr=attr, method=method, sourcename=fba_sourcename,
            download_FBA_if_missing=kwargs['download_FBA_if_missing'])

    return fba_wsec
