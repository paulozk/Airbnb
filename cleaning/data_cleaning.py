import pandas as pd
import pdpipe as pdp
import numpy as np
import sys
import time
import geopandas as gpd

# define a custom adhocstage subclass that allows for easy argument passing to transform function
class AdHogStageArg(pdp.AdHocStage):
    """An ad-hoc stage of a pandas DataFrame-processing pipeline.

    Parameters
    ----------
    transform : callable
        The transformation this stage applies to dataframes.
    prec : callable, default None
        A callable that returns a boolean value. Represent a a precondition
        used to determine whether this stage can be applied to a given
        dataframe. If None is given, set to a function always returning True.
    """

    def __init__(self, transform, prec=None, **kwargs):
        if prec is None:
            prec = True
        self._adhoc_transform = transform
        self._adhoc_prec = prec
        super().__init__(transform)

        self.kwargs = kwargs

    def _prec(self, df):
        return self._adhoc_prec

    def _transform(self, df, verbose):
        try:
            return self._adhoc_transform(df, **self.kwargs)
        except TypeError:
            return self._adhoc_transform(df)


def add_neighbourhood(data, **kwargs):
    start_time = time.time()

    # append column to data with its geometry
    data['geometry'] = gpd.points_from_xy(data.longitude.astype(float), data.latitude.astype(float))
    data['neighbourhood'] = np.empty(len(data), dtype='str')
    n_adam = 0

    for i, neighbourhood in enumerate(kwargs['geo_data'].neighbourhood):
        # get the indices of points belonging to neighborhood i
        idx = np.where(data['geometry'].map(lambda x: x.within(kwargs['geo_data'].geometry[i])))
        n_adam += len(idx[0])
        data['neighbourhood'].iloc[idx] = neighbourhood

    time_elapsed = time.time() - start_time
    print("add_neighbourhood:", time_elapsed)

    return data


# define a function that, given some columns of the dataframe, drops rows of duplicates
# SLOW!
def get_duplicate_rows(data):
    start_time = time.time()

    data = data.reset_index(drop=True)
    column_names = ['id', 'listing_url']
    rows_to_drop = np.zeros(0, dtype=int)
    for column_name in column_names:
        # remove rows with the same values, but keep one, obviously
        # get indices of rows where the listing_url occurs in at least one more row
        indices_duplicates = np.where(data[column_name].dropna().duplicated())[0]

        # each index specifies a value that occurs at least twice; drop all but 1 row containing such URL,
        # for each duplicate index
        for index in indices_duplicates:
            # for current index, find all rows with same URL
            index_duplicates = np.where(data[column_name] == data[column_name].iloc[index])[0]
            # keep one of the rows, remove the rest
            rows_to_drop = np.concatenate([rows_to_drop, index_duplicates[1:]])

    # rows_to_drop = data.index[rows_to_drop]
    result = data.drop(rows_to_drop, axis=0)

    time_elapsed = time.time() - start_time
    print("get_duplicate_rows:", time_elapsed)

    return result

# given columns containing a list of values, convert this column to multiple binary columns; one for each possible
# value
# example:
# original column -
# [a, b, c, d, e]
# [e, f]
# new -
# [0-1] [0-1] [0-1] [0-1] [0-1] [0-1]
# VERY SLOW!
def expand_columns(data):
    start_time = time.time()

    data = data.reset_index(drop=True)
    columns = ['host_verifications', 'amenities']
    for col in columns:
        column = data[col]
        possible_values = []
        # get unique values
        unique_values = column.unique()
        # drop NaNs
        unique_values = pd.Series(unique_values).dropna()
        unique_values = unique_values.str.replace('[\'\[\]\{\}]', '')
        # print(unique_values)
        for value_set in unique_values:
            # values = value_set.replace('')
            values = value_set.split(',')
            for value in values:
                if (value not in possible_values):
                    possible_values.append(value)

        # print("Possible values:", possible_values)
        # after getting all the possible values, create binary columns for each of them and fill them accordingly
        new_columns = np.zeros((column.shape[0], len(possible_values)))
        # temporarily remove NaNs from column
        column_nonan = column.dropna()
        # set appropriate columns 1 if they occur
        for i, value in enumerate(possible_values):
            # print("----------------")
            # print(value)
            # print(np.where(column_nonan.str.contains(value))[0])
            # set binary values at in the right rows and at the right column to 1
            # print(value)
            idx_value = np.where(column_nonan.str.contains(value))[0]
            new_columns[idx_value, i] = 1.0

        # set all missing values to NaNs in all new columns
        new_columns[column.isnull()] = np.nan
        # convert numpy array to pandas dataframe with the appropriate names
        possible_values = pd.Series(possible_values).map(lambda x: column.name + '_' + x)
        new_columns = pd.DataFrame(data=new_columns, columns=possible_values)
        # append new columns to dataframe
        data = pd.concat([data, new_columns], axis=1)
        # drop source column
        data = data.drop(col, axis=1)

    time_elapsed = time.time() - start_time
    print("expand_columns:", time_elapsed)

    return data

# Drop rows that do not contain certain jurisdiction names
def filter_on_jurisdiction():
    start_time = time.time()

    keep_list = ['{Amsterdam," NL"}',
                 '{Amsterdam," NL Zip Codes 2"," Amsterdam"," NL"}',
                 '{Amsterdam}',
                 '{Amsterdam," NL Zip Codes 2"}',
                 np.nan]

    func = {'jurisdiction_names': lambda x: x not in keep_list}
    result = pdp.RowDrop(func)

    time_elapsed = time.time() - start_time
    print("Filter_on_jurisdiction:", time_elapsed)

    return result

def filter_on_neighbourhood():
    start_time = time.time()

    # drop rows with empty neighbourhood -> not in Amsterdam!
    func = {'neighbourhood': lambda x: x == ''}
    result = pdp.RowDrop(func)

    time_elapsed = time.time() - start_time
    print("filter_on_neighbourhood:", time_elapsed)

    return result

# uniformize missing values by replacing them by NaNs
def uniformize_missing(columns):
    start_time = time.time()
    missing_values_strings = ['NaN', '??', '*', 'UNK', '-', '###']
    func = lambda x: np.nan if x in missing_values_strings else x

    time_elapsed = time.time() - start_time
    result = pdp.ApplyByCols(columns, func)
    print("uniformize_missing:", time_elapsed)

    return result

# uniformize boolean values by replacing them by 1 (True) or 0 (False)
def uniformize_boolean(columns):
    start_time = time.time()
    true_strings = ['t', 'true', 'yes', 'y', True]
    false_strings = ['f', 'false', 'n', 'no', False]

    func_true = lambda x: 1.0 if x in true_strings else x
    func_false = lambda x: 0.0 if x in false_strings else x

    result = pdp.ApplyByCols(columns, func_true) + pdp.ApplyByCols(columns, func_false)

    time_elapsed = time.time() - start_time
    print("uniformize_boolean:", time_elapsed)

    return result

# define a function that unifies the data formats into one format
def unify_datetimes(data):
    start_time = time.time()

    column_names = ['host_since', 'first_review', 'last_review']
    for column_name in column_names:
        # identify rows with unix timestamps
        idx_timestamp = data[column_name].str.extract('(\d\d\d\d\d\d\d\d)').dropna().index
        # convert unix timestamps to standard format
        dates_timestamp = pd.to_datetime(data[column_name][idx_timestamp], unit='s').dt.strftime('%Y-%m-%d')

        idx_normal = np.setdiff1d(data.index, idx_timestamp)
        dates_normal = pd.to_datetime(data[column_name][idx_normal]).dt.strftime('%Y-%m-%d')

        # join both date columns together and return
        data[column_name] = pd.concat([dates_timestamp, dates_normal])

    time_elapsed = time.time() - start_time
    print("unify_datetimes:", time_elapsed)

    return data

# convert monetary values to float by stripping the '$' and casting to numeric value
def uniformize_monetary():
    start_time = time.time()

    monetary_columns = ['price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee', 'extra_people']

    func = lambda x: float(x[1:].replace(',', '')) if type(x) == str else x

    result = pdp.ApplyByCols(monetary_columns, func)

    time_elapsed = time.time() - start_time
    print("uniformize_monetary:", time_elapsed)

    return result

# convert percentage values to float by stripping the '%' and casting to numeric value
def uniformize_percentage():
    start_time = time.time()

    percentage_columns = ['host_response_rate']

    func = lambda x: float(x[:-1]) if type(x) == str else x

    result = pdp.ApplyByCols(percentage_columns, func)

    time_elapsed = time.time() - start_time
    print("uniformize_percentage:", time_elapsed)

    return result

# drop columns that only contain either 1 possible values or NaNs
def drop_useless():
    start_time = time.time()

    useless_column = ['experiences_offered', 'has_availability', 'requires_license', 'is_business_travel_ready']

    result = pdp.ColDrop(useless_column)

    time_elapsed = time.time() - start_time
    print("drop_useless:", time_elapsed)

    return result

# build the data cleaning pipeline and return the built pipeline object
def build_pipeline(column_names, data_geo):
    print("Now building pipeline...")
    # initialize the pipeline
    pipeline = filter_on_jurisdiction()
    # get rid of duplicates
    pipeline += pdp.AdHocStage(transform=get_duplicate_rows)
    # uniformize missing values
    pipeline += uniformize_missing(column_names)
    # add neighbourhood of data points based on latitude and longitude
    kwargs = {'geo_data': data_geo}
    pipeline += AdHogStageArg(transform=add_neighbourhood, **kwargs)
    # drop rows that do not belong to Amsterdam, according to lat lon
    pipeline += filter_on_neighbourhood()
    # uniformize boolean values
    pipeline += uniformize_boolean(column_names)
    # uniformize datetimes
    pipeline += pdp.AdHocStage(transform=unify_datetimes)
    # uniformize monetary values and convert to numeric
    pipeline += uniformize_monetary()
    # uniformize percentage values and convert to numeric
    pipeline += uniformize_percentage()
    # add new boolean columns for values within certain columns
    pipeline += pdp.AdHocStage(transform=expand_columns)
    # drop columns that contain no useful information
    pipeline += drop_useless()

    return pipeline

# clean the dataset located at path_in and store the cleaned data at location path_out
def clean_dataset(path_in, path_out, json_path):
    # read in the data from path_in
    data = pd.read_csv(path_in)
    print("Starting data shape:", data.shape)
    data_geo = gpd.read_file(json_path)
    # build the data cleaning pipeline
    pipeline = build_pipeline(data.columns, data_geo)
    print("Cleaning data with pipeline..")
    # clean data with pipeline
    data_cleaned = pipeline.apply(data, verbose=False)
    print("Cleaned data shape:", data_cleaned.shape)
    # store the data to path_out
    data_cleaned.to_csv(path_out)

def clean_dataset_json(df_json):
    # read in the data from path_in
    data = pd.read_json(df_json)
    # build the data cleaning pipeline
    pipeline = build_pipeline(data.columns)
    print("Cleaning data with pipeline..")
    # clean data with pipeline
    data_cleaned = pipeline.apply(data, verbose=False)
    # return jsonified version of cleaned dataset
    data_cleaned_json = data_cleaned.to_json()

    return data_cleaned_json

def clean_dataset_csv(data, data_geo):
    # build the data cleaning pipeline
    pipeline = build_pipeline(column_names=data.columns, data_geo=data_geo)
    print("Cleaning data with pipeline..")
    # clean data with pipeline
    data_cleaned = pipeline.apply(data, verbose=False)
    # return jsonified version of cleaned dataset
    #data_cleaned_json = data_cleaned.to_json()

    return data_cleaned




if(__name__ == '__main__'):
    try:
        _, path_in, path_out, path_json = sys.argv
    except:
        print("Please supply input and output path!")
        sys.exit()

    clean_dataset(path_in, path_out, path_json)