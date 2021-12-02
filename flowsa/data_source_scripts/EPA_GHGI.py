# EPA_GHGI.py (flowsa)
# !/usr/bin/env python3
# coding=utf-8
"""
Inventory of US EPA GHG
https://www.epa.gov/ghgemissions/inventory-us-greenhouse-gas-emissions-and-sinks-1990-2018
"""

import io
import zipfile
import yaml
import numpy as np
import pandas as pd
from flowsa.flowbyfunctions import assign_fips_location_system
from flowsa.settings import log, datapath

DEFAULT_YEAR = 9999

# Decided to add tables as a constant in the source code because
# the YML config isn't available in the ghg_call method.
# Only keeping years 2010-2018 for the following tables:
sourcefile = datapath + 'GHGI_tables.yaml'
with open(sourcefile, 'r') as f:
     table_dict = yaml.safe_load(f)

A_17_COMMON_HEADERS = ['Res.', 'Comm.', 'Ind.', 'Trans.', 'Elec.', 'Terr.', 'Total']
A_17_TBTU_HEADER = ['Adjusted Consumption (TBtu)a', 'Adjusted Consumption (TBtu)']
A_17_CO2_HEADER = ['Emissionsb (MMT CO2 Eq.) from Energy Use',
                   'Emissions (MMT CO2 Eq.) from Energy Use']

SPECIAL_FORMAT = ["3-10", "3-22", "4-46", "4-50", "4-80", "A-17", "A-93", "A-94", "A-118", "5-29"]
SRC_NAME_SPECIAL_FORMAT = ["T_3_22", "T_4_43", "T_4_80", "T_A_17"]
Activity_Format_A = ["T_5_30", "T_A_17", "T_ES_5"]
Activity_Format_B = ["T_2_1", "T_3_21", "T_3_22", "T_4_48", "T_5_18"]

DROP_COLS = ["Unnamed: 0", "1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998",
             "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009"]

YEARS = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019"]


def ghg_url_helper(build_url, config, args):
    """
    This helper function uses the "build_url" input from flowbyactivity.py,
    which is a base url for data imports that requires parts of the url text
    string to be replaced with info specific to the data year. This function
    does not parse the data, only modifies the urls from which data is
    obtained.
    :param build_url: string, base url
    :param config: dictionary, items in FBA method yaml
    :param args: dictionary, arguments specified when running flowbyactivity.py
        flowbyactivity.py ('year' and 'source')
    :return: list, urls to call, concat, parse, format into Flow-By-Activity
        format
    """
    annex_url = config['url']['annex_url']
    return [build_url, annex_url]


def fix_a17_headers(header):
    """
    Fix A-17 headers, trim white spaces, convert shortened words such as Elec., Res., etc.
    :param header: str, column header
    :return: str, modified column header
    """
    if header == A_17_TBTU_HEADER[0]:
        header = f' {A_17_TBTU_HEADER[1].strip()}'.replace('')
    elif header == A_17_CO2_HEADER[0]:
        header = f' {A_17_CO2_HEADER[1].strip()}'
    else:
        header = header.strip()
        header = header.replace('Res.', 'Residential')
        header = header.replace('Comm.', 'Commercial')
        header = header.replace('Ind.', 'Industrial Other')
        header = header.replace('Trans.', 'Transportation')
        header = header.replace('Elec.', 'Electricity Power')
        header = header.replace('Terr.', 'U.S. Territory')
    return header


def cell_get_name(value, default_flow_name):
    """
    Given a single string value (cell), separate the name and units.
    :param value: str
    :param default_flow_name: indicate return flow name string subset
    :return: flow name for row
    """
    if '(' not in value:
        return default_flow_name.replace('__type__', value.strip())

    spl = value.split(' ')
    name = ''
    found_units = False
    for sub in spl:
        if '(' not in sub and not found_units:
            name = f'{name.strip()} {sub}'
        else:
            found_units = True
    return default_flow_name.replace('__type__', name.strip())


def cell_get_units(value, default_units):
    """
    Given a single string value (cell), separate the name and units.
    :param value: str
    :param default_units: indicate return units string subset
    :return: unit for row
    """
    if '(' not in value:
        return default_units

    spl = value.split(' ')
    name = ''
    found_units = False
    for sub in spl:
        if ')' in sub:
            found_units = False
        if '(' in sub or found_units:
            name = f'{name} {sub.replace("(", "").replace(")", "")} '
            found_units = True
    return name.strip()


def series_separate_name_and_units(series, default_flow_name, default_units):
    """
    Given a series (such as a df column), split the contents' strings into a name and units.
    An example might be converting "Carbon Stored (MMT C)" into ["Carbon Stored", "MMT C"].

    :param series: df column
    :param default_flow_name: df column for flow name to be modified
    :param default_units: df column for units to be modified
    :return: str, flowname and units for each row in df
    """
    names = series.apply(lambda x: cell_get_name(x, default_flow_name))
    units = series.apply(lambda x: cell_get_units(x, default_units))
    return {'names': names, 'units': units}


def ghg_call(url, response_load, args):
    """
    Convert response for calling url to pandas dataframe, begin parsing df
    into FBA format
    :param url: string, url
    :param r: df, response from url call
    :param args: dictionary, arguments specified when running
        flowbyactivity.py ('year' and 'source')
    :return: pandas dataframe of original source data
    """
    df = None
    year = args['year']
    with zipfile.ZipFile(io.BytesIO(response_load.content), "r") as f:
        frames = []
        if 'annex' in url:
            is_annex = True
            t_tables = table_dict['Annex']
        else:
            is_annex = False
            t_tables = table_dict['Tables']
        for chapter, tables in t_tables.items():
            for table in tables:
                # path = os.path.join("Chapter Text", chapter, f"Table {table}.csv")
                if is_annex:
                    path = f"Annex/Table {table}.csv"
                else:
                    path = f"Chapter Text/{chapter}/Table {table}.csv"
                data = f.open(path)
                if table not in SPECIAL_FORMAT:
                    df = pd.read_csv(data, skiprows=2, encoding="ISO-8859-1", thousands=",")
                elif '3-' in table:
                    if table == '3-10':
                        df = pd.read_csv(data, skiprows=1, encoding="ISO-8859-1",
                                         thousands=",", decimal=".")
                    else:
                        # Skip first two rows, as usual, but make headers the next 3 rows:
                        df = pd.read_csv(data, skiprows=2, encoding="ISO-8859-1",
                                         header=[0, 1, 2], thousands=",")
                        # The next two rows are headers and the third is units:
                        new_headers = []
                        for col in df.columns:
                            # unit = col[2]
                            new_header = 'Unnamed: 0'
                            if 'Unnamed' not in col[0]:
                                if 'Unnamed' not in col[1]:
                                    new_header = f'{col[0]} {col[1]}'
                                else:
                                    new_header = col[0]
                                if 'Unnamed' not in col[2]:
                                    new_header += f' {col[2]}'
                                # unit = col[2]
                            elif 'Unnamed' in col[0] and 'Unnamed' not in col[2]:
                                new_header = col[2]
                            new_headers.append(new_header)
                        df.columns = new_headers
                elif '4-' in table:
                    if table == '4-46':
                        df = pd.read_csv(data, skiprows=1, encoding="ISO-8859-1",
                                         thousands=",", decimal=".")
                    else:
                        df = pd.read_csv(data, skiprows=2, encoding="ISO-8859-1",
                                     thousands=",", decimal=".")
                elif 'A-' in table:
                    if table == 'A-17':
                        # A-17  is similar to T 3-23, the entire table is 2012 and
                        # headings are completely different.
                        if str(year) == '2013':
                            df = pd.read_csv(data, skiprows=2, encoding="ISO-8859-1",
                                             header=[0, 1], thousands=",")
                            new_headers = []
                            header_grouping = ''
                            for col in df.columns:
                                if 'Unnamed' in col[0]:
                                    # new_headers.append(f'{header_grouping}{col[1]}')
                                    new_headers.append(f'{fix_a17_headers(col[1])}'
                                                       f'{header_grouping}')
                                else:
                                    if len(col) == 2:
                                        # header_grouping = f'{col[0]}__'
                                        if col[0] == A_17_TBTU_HEADER[0]:
                                            header_grouping = f' {A_17_TBTU_HEADER[1].strip()}'
                                        else:
                                            header_grouping = f' {A_17_CO2_HEADER[1].strip()}'
                                    # new_headers.append(f'{header_grouping}{col[1]}')
                                    new_headers.append(f'{fix_a17_headers(col[1])}'
                                                       f'{header_grouping}')
                            df.columns = new_headers
                            nan_col = 'Electricity Power Emissions (MMT CO2 Eq.) from Energy Use'
                            fill_col = 'Unnamed: 12_level_1 Emissions (MMT CO2 Eq.) from Energy Use'
                            df = df.drop(nan_col, 1)
                            df.columns = [nan_col if x == fill_col else x for x in df.columns]
                            df['Year'] = year
                    else:
                        df = pd.read_csv(data, skiprows=1, encoding="ISO-8859-1",
                                         thousands=",", decimal=".")
                elif '5-' in table:
                    df = pd.read_csv(data, skiprows=1, encoding="ISO-8859-1",
                                     thousands=",", decimal=".")

                if df is not None and len(df.columns) > 1:
                    years = YEARS.copy()
                    years.remove(str(year))
                    df = df.drop(columns=(DROP_COLS + years), errors='ignore')
                    # Assign SourceName now while we still have access to the table name:
                    source_name = f"EPA_GHGI_T_{table.replace('-', '_')}"
                    df["SourceName"] = source_name
                    frames.append(df)

        # return pd.concat(frames)
        return frames


def get_unnamed_cols(df):
    """
    Get a list of all unnamed columns, used to drop them.
    :param df: df being formatted
    :return: list, unnamed columns
    """
    return [col for col in df.columns if "Unnamed" in col]


def get_table_meta(source_name):
    """Find and return table meta from source_name."""
    if "_A_" in source_name:
        td = table_dict['Annex']
    else:
        td = table_dict['Tables']
    for chapter in td.keys():
        for k, v in td[chapter].items():
            if k.replace("-","_") in source_name:
                return v

def is_consumption(source_name):
    """
    Determine whether the given source contains consumption or production data.
    :param source_name: df
    :return: True or False
    """
    if 'consum' in get_table_meta(source_name)['desc'].lower():
        return True
    return False

def strip_char(text):
    """
    Removes the footnote chars from the text
    """
    text = text + " "
    notes = [" a ", " b ", " c ", " d ", " e ", " f ", " g ", " h ", " i ", " j ", " k "]
    for i in notes:
        if i in text:
            text_split = text.split(i)
            text = text_split[0]
    return text.strip()


def ghg_parse(dataframe_list, args):
    """
    Combine, parse, and format the provided dataframes
    :param dataframe_list: list of dataframes to concat and format
    :param args: dictionary, used to run flowbyactivity.py
        ('year' and 'source')
    :return: df, parsed and partially formatted to flowbyactivity
        specifications
    """
    cleaned_list = []
    for df in dataframe_list:
        special_format = False
        source_name = df["SourceName"][0]
        log.info('Processing Source Name %s', source_name)
        for src in SRC_NAME_SPECIAL_FORMAT:
            if src in source_name:
                special_format = True

        # Specify to ignore errors in case one of the drop_cols is missing.
        drop_cols = get_unnamed_cols(df)
        df = df.drop(columns=drop_cols, errors='ignore')
        is_cons = is_consumption(source_name)

        if not special_format or "T_4_" not in source_name:
            # Rename the PK column from data_type to "ActivityProducedBy" or "ActivityConsumedBy":
            if is_cons:
                df = df.rename(columns={df.columns[0]: "ActivityConsumedBy"})
                df["ActivityProducedBy"] = 'None'
            else:
                df = df.rename(columns={df.columns[0]: "ActivityProducedBy"})
                df["ActivityConsumedBy"] = 'None'
        else:
            df["ActivityConsumedBy"] = 'None'
            df["ActivityProducedBy"] = 'None'


        df["FlowType"] = "ELEMENTARY_FLOW"
        df["Location"] = "00000"

        id_vars = ["SourceName", "ActivityConsumedBy", "ActivityProducedBy", "FlowType", "Location"]
        if special_format and "Year" in df.columns:
            id_vars.append("Year")
            # Cast Year column to numeric and delete any years != year
            df = df[pd.to_numeric(df["Year"], errors="coerce") == int(args['year'])]

        # Set index on the df:
        df.set_index(id_vars)
        switch_year_apb = ["EPA_GHGI_T_4_14", "EPA_GHGI_T_4_50"]
        if special_format:
            if "T_4_" not in source_name:
                df = df.melt(id_vars=id_vars, var_name="FlowName", value_name="FlowAmount")
            else:
                df = df.melt(id_vars=id_vars, var_name="Units", value_name="FlowAmount")
        else:
            df = df.melt(id_vars=id_vars, var_name="Year", value_name="FlowAmount")
            if source_name in switch_year_apb:
                df = df.rename(columns={'ActivityProducedBy': 'Year', 'Year': 'ActivityProducedBy'})


        # Dropping all rows with value "+"
        try:
            df = df[~df["FlowAmount"].str.contains("\\+", na=False)]
        except AttributeError as ex:
            log.info(ex)
        # Dropping all rows with value "NE"
        try:
            df = df[~df["FlowAmount"].str.contains("NE", na=False)]
        except AttributeError as ex:
            log.info(ex)

        # Convert all empty cells to nan cells
        df["FlowAmount"].replace("", np.nan, inplace=True)
        # Table 3-10 has some NO values, dropping these.
        df["FlowAmount"].replace("NO", np.nan, inplace=True)
        # Table A-118 has some IE values, dropping these.
        df["FlowAmount"].replace("IE", np.nan, inplace=True)

        # Drop any nan rows
        df.dropna(subset=['FlowAmount'], inplace=True)

        df["Description"] = 'None'
        df["Unit"] = "Other"

        # Update classes:
        meta = get_table_meta(source_name)
        if source_name == "EPA_GHGI_T_3_21" and int(args["year"]) < 2015:
            # skip don't do anything: The lines are blank
            print("There is no data for this year and source")
        else:
            df.loc[df["SourceName"] == source_name, "Class"] = meta["class"]
            df.loc[df["SourceName"] == source_name, "Unit"] = meta["unit"]
            df.loc[df["SourceName"] == source_name, "Description"] = meta["desc"]
            df.loc[df["SourceName"] == source_name, "Compartment"] = meta["compartment"]
            if not special_format or "T_4_" in source_name:
                df.loc[df["SourceName"] == source_name, "FlowName"] = meta["activity"]
            else:
                if "T_4_" not in source_name:
                    flow_name_units = series_separate_name_and_units(df["FlowName"],
                                                                     meta["activity"],
                                                                     meta["unit"])
                    df['Unit'] = flow_name_units['units']
                    df.loc[df["SourceName"] == source_name, "FlowName"] = flow_name_units['names']

        # We also need to fix the Activity PRODUCED or CONSUMED, now that we know units.
        # Any units TBtu will be CONSUMED, all other units will be PRODUCED.
        if is_cons:
            df['ActivityProducedBy'] = df['ActivityConsumedBy']
            df.loc[df["Unit"] == 'TBtu', 'ActivityProducedBy'] = 'None'
            df.loc[df["Unit"] != 'TBtu', 'ActivityConsumedBy'] = 'None'
        else:
            df['ActivityConsumedBy'] = df['ActivityProducedBy']
            df.loc[df["Unit"] == 'TBtu', 'ActivityProducedBy'] = 'None'
            df.loc[df["Unit"] != 'TBtu', 'ActivityConsumedBy'] = 'None'

        if 'Year' not in df.columns:
          #  df['Year'] = meta.get("year", DEFAULT_YEAR)
          df['Year'] = args['year']

        if source_name == "EPA_GHGI_T_4_33":
            df = df.rename(columns={'Year': 'ActivityProducedBy', 'ActivityProducedBy': 'Year'})
        year_int = ["EPA_GHGI_T_4_33", "EPA_GHGI_T_4_50"]
        # Some of the datasets, 4-43 and 4-80, still have years we don't want at this point.
        # Remove rows matching the years we don't want:
        try:

            if source_name in year_int:
                df = df[df['Year'].isin([int(args['year'])])]
            else:
                df = df[df['Year'].isin([args['year']])]



        except AttributeError as ex:
            log.info(ex)

        # Add DQ scores
        df["DataReliability"] = 5  # tmp
        df["DataCollection"] = 5  # tmp
        # Fill in the rest of the Flow by fields so they show "None" instead of nan.76i
        df["MeasureofSpread"] = 'None'
        df["DistributionType"] = 'None'
        df["LocationSystem"] = 'None'

        df = assign_fips_location_system(df, str(args['year']))
        modified_activity_list = ["EPA_GHGI_T_ES_5"]
        multi_chem_names = ["EPA_GHGI_T_2_1", "EPA_GHGI_T_4_46", "EPA_GHGI_T_5_7", "EPA_GHGI_T_5_29", "EPA_GHGI_T_ES_5"]
        source_No_activity = ["EPA_GHGI_T_3_22"]
        source_activity_1 = ["EPA_GHGI_T_3_8", "EPA_GHGI_T_3_9", "EPA_GHGI_T_3_14", "EPA_GHGI_T_3_15",
                             "EPA_GHGI_T_5_3", "EPA_GHGI_T_5_18", "EPA_GHGI_T_5_19", "EPA_GHGI_T_A_76",
                             "EPA_GHGI_T_A_77", "EPA_GHGI_T_3_10"]
        source_activity_2 =  ["EPA_GHGI_T_3_38", "EPA_GHGI_T_3_63"]
        double_activity = ["EPA_GHGI_T_4_48"]
        note_par = ["EPA_GHGI_T_4_14", "EPA_GHGI_T_4_99"]
        if source_name in multi_chem_names:
            bool_apb = False
            apbe_value = ""
            flow_name_list = ["CO2", "CH4", "N2O", "NF3", "HFCs", "PFCs", "SF6", "NF3", "CH4 a", "N2O b", "CO", "NOx"]
            for index, row in df.iterrows():
                apb_value = row["ActivityProducedBy"]
                if "CH4" in apb_value:
                    apb_value = "CH4"
                elif "N2O" in apb_value:
                    apb_value = "N2O"
                elif "CO2" in apb_value:
                    apb_value = "CO2"

                if apb_value in flow_name_list:
                    apbe_value = apb_value
                    df.loc[index, 'FlowName'] = apbe_value
                    df.loc[index, 'ActivityProducedBy'] = "All activities"
                    bool_apb = True
                else:
                    if bool_apb == True:
                        apb_txt = df.loc[index, 'ActivityProducedBy']
                        apb_txt = strip_char(apb_txt)
                        df.loc[index, 'ActivityProducedBy'] = apb_txt
                        df.loc[index, 'FlowName'] = apbe_value
                    else:
                        apb_txt = df.loc[index, 'ActivityProducedBy']
                        apb_txt = strip_char(apb_txt)
                        df.loc[index, 'ActivityProducedBy'] = apb_txt

                if "Total" == apb_value or "Total " == apb_value:
                  df = df.drop(index)
            if source_name == "EPA_GHGI_T_ES_5":
                df = df.rename(columns={'FlowName': 'ActivityProducedBy', 'ActivityProducedBy': 'FlowName'})
        elif source_name in source_No_activity:
            bool_apb = False
            apbe_value = ""
            flow_name_list = ["Industry", "Transportation", "U.S. Territories"]
            for index, row in df.iterrows():
                unit = row["Unit"]
                if unit.strip() == "MMT  CO2":
                        df.loc[index, 'Unit'] = "MMT CO2e"
                if df.loc[index, 'Unit'] != "MMT CO2e":
                    df = df.drop(index)
                else:
                    apb_value = row["ActivityProducedBy"]
                    if apb_value in flow_name_list:
                        apbe_value = apb_value
                        if apb_value == "U.S. Territories":
                            df.loc[index, 'Location'] = "99000"
                        df.loc[index, 'FlowName'] = "CO2"
                        df.loc[index, 'ActivityProducedBy'] = apbe_value + " " + "All activities"
                        bool_apb = True
                    else:
                        if bool_apb == True:
                            df.loc[index, 'FlowName'] = "CO2"
                            apb_txt = df.loc[index, 'ActivityProducedBy']
                            apb_txt = strip_char(apb_txt)
                            if apbe_value == "U.S. Territories":
                                df.loc[index, 'Location'] = "99000"
                            df.loc[index, 'ActivityProducedBy'] = apbe_value + " " + apb_txt
                        else:
                            apb_txt = df.loc[index, 'ActivityProducedBy']
                            apb_txt = strip_char(apb_txt)
                            df.loc[index, 'ActivityProducedBy'] = apbe_value + " " + apb_txt
                        if "Total" == apb_value or "Total " == apb_value:
                            df = df.drop(index)
        elif source_name in source_activity_1:
            bool_apb = False
            apbe_value = ""
            flow_name_list = ["Electric Power", "Industrial", "Commercial", "Residential", "U.S. Territories",
                              "U.S. Territories a", "Transportation",
                              "Fuel Type/Vehicle Type a", "Diesel On-Road b", "Alternative Fuel On-Road", "Non-Road c",
                              "Gasoline On-Road b", "Non-Road", "Exploration a", "Production (Total)",
                              "Crude Oil Transportation", "Refining", "Exploration b", "Cropland", "Grassland"]
            for index, row in df.iterrows():
                apb_value = row["ActivityProducedBy"]
                start_activity = row["FlowName"]
                if apb_value in flow_name_list:
                    if "U.S. Territories" in apb_value:
                        df.loc[index, 'Location'] = "99000"
                    elif "U.S. Territories" in apbe_value:
                        df.loc[index, 'Location'] = "99000"
                    apbe_value = apb_value
                    apbe_value = strip_char(apbe_value)
                    df.loc[index, 'FlowName'] = start_activity
                    df.loc[index, 'ActivityProducedBy'] = "All activities" + " " + apbe_value
                    bool_apb = True
                else:
                    if bool_apb == True:
                        if "U.S. Territories" in apb_value:
                            df.loc[index, 'Location'] = "99000"
                        elif "U.S. Territories" in apbe_value:
                            df.loc[index, 'Location'] = "99000"
                        df.loc[index, 'FlowName'] = start_activity
                        apb_txt = df.loc[index, 'ActivityProducedBy']
                        apb_txt = strip_char(apb_txt)
                        df.loc[index, 'ActivityProducedBy'] = apb_txt + " " + apbe_value
                        if source_name == "EPA_GHGI_T_3_10":
                            df.loc[index, 'FlowName'] = apb_txt
                    else:
                        if "U.S. Territories" in apb_value:
                            df.loc[index, 'Location'] = "99000"
                        elif "U.S. Territories" in apbe_value:
                            df.loc[index, 'Location'] = "99000"
                        apb_txt = df.loc[index, 'ActivityProducedBy']
                        apb_txt = strip_char(apb_txt)
                        apb_final = apb_txt + " " + apbe_value
                        df.loc[index, 'ActivityProducedBy'] = apb_final.strip()
                if "Total" == apb_value or "Total " == apb_value:
                  df = df.drop(index)
        elif source_name in source_activity_2:
            bool_apb = False
            apbe_value = ""
            flow_name_list = ["Explorationb", "Production", "Processing", "Transmission and Storage", "Distribution",
                              "Crude Oil Transportation", "Refining", "Exploration" ]
            for index, row in df.iterrows():
                apb_value = row["ActivityProducedBy"]
                start_activity = row["FlowName"]
                if apb_value.strip() in flow_name_list:
                    apbe_value = apb_value
                    if apbe_value == "Explorationb":
                        apbe_value = "Exploration"
                    df.loc[index, 'FlowName'] = start_activity
                    df.loc[index, 'ActivityProducedBy'] = apbe_value
                    bool_apb = True
                else:
                    if bool_apb == True:
                        df.loc[index, 'FlowName'] = start_activity
                        apb_txt = df.loc[index, 'ActivityProducedBy']
                        apb_txt = strip_char(apb_txt)
                        if apb_txt == "Gathering and Boostingc":
                            apb_txt = "Gathering and Boosting"
                        df.loc[index, 'ActivityProducedBy'] = apbe_value + " - " + apb_txt
                    else:
                        apb_txt = df.loc[index, 'ActivityProducedBy']
                        apb_txt = strip_char(apb_txt)
                        df.loc[index, 'ActivityProducedBy'] = apb_txt + " " + apbe_value
                if "Total" == apb_value or "Total " == apb_value:
                  df = df.drop(index)
        elif source_name in double_activity:
            for index, row in df.iterrows():
                df.loc[index, 'FlowName'] = df.loc[index, 'ActivityProducedBy']
        else:
            if source_name in "EPA_GHGI_T_4_80":
                for index, row in df.iterrows():
                    df.loc[index, 'FlowName'] = df.loc[index, 'Units']
                    df.loc[index, 'ActivityProducedBy'] = "Aluminum Production"
            elif source_name in "EPA_GHGI_T_4_94":
                for index, row in df.iterrows():
                    df.loc[index, 'FlowName'] = df.loc[index, 'ActivityProducedBy']
                    df.loc[index, 'ActivityProducedBy'] = "Electronics Production"
            elif source_name in "EPA_GHGI_T_4_99":
                for index, row in df.iterrows():
                    df.loc[index, 'FlowName'] = df.loc[index, 'ActivityProducedBy']
                    df.loc[index, 'ActivityProducedBy'] = "ODS Substitute"
            elif source_name in "EPA_GHGI_T_4_33":
                for index, row in df.iterrows():
                    df.loc[index, 'Unit'] = df.loc[index, 'ActivityProducedBy']
                    df.loc[index, 'ActivityProducedBy'] = "Caprolactam Production"
            elif source_name in "EPA_GHGI_T_A_101":
                for index, row in df.iterrows():
                    apb_value = strip_char(row["ActivityProducedBy"])
                    df.loc[index, 'ActivityProducedBy'] = apb_value
            elif source_name == "EPA_GHGI_T_4_50":
                for index, row in df.iterrows():
                    apb_value = strip_char(row["ActivityProducedBy"])
                    df.loc[index, 'ActivityProducedBy'] = "HFC-23 Production"
                    if "kt" in apb_value:
                        df.loc[index, 'Unit'] = "kt"
                    else:
                        df.loc[index, 'Unit'] = "MMT CO2e"
            elif source_name in note_par:
                for index, row in df.iterrows():
                    apb_value = strip_char(row["ActivityProducedBy"])
                    if "(" in apb_value:
                        text_split = apb_value.split("(")
                        df.loc[index, 'ActivityProducedBy'] = text_split[0]
            else:
                for index, row in df.iterrows():
                    if "CO2" in df.loc[index, 'Unit']:
                        df.loc[index, 'Unit'] = "MMT CO2e"
                    if "U.S. Territory" in df.loc[index, 'ActivityProducedBy']:
                        df.loc[index, 'Location'] = "99000"

            df.drop(df.loc[df['ActivityProducedBy'] == "Total"].index, inplace=True)
            df.drop(df.loc[df['ActivityProducedBy'] == "Total "].index, inplace=True)
            df.drop(df.loc[df['FlowName'] == "Total"].index, inplace=True)
            df.drop(df.loc[df['FlowName'] == "Total "].index, inplace=True)


        if source_name in modified_activity_list:

            if is_cons:
                df = df.rename(columns={'FlowName': 'ActivityConsumedBy', 'ActivityConsumedBy': 'FlowName'})
            else:
                df = df.rename(columns={'FlowName': 'ActivityProducedBy', 'ActivityProducedBy': 'FlowName'})
           # if source_name == "EPA_GHGI_T_2_1":
           #     df["FlowName"] = "CO2 eq"

        df = df.loc[:, ~df.columns.duplicated()]
        cleaned_list.append(df)

    if cleaned_list:
        for df in cleaned_list:
            # Remove commas from numbers again in case any were missed:
            df["FlowAmount"].replace(',', '', regex=True, inplace=True)
        return cleaned_list
        # df = pd.concat(cleaned_list)
    else:
        df = pd.DataFrame()
        return df
