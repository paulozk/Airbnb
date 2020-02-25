import pandas as pd
import pdpipe as pdp
import numpy as np

# define a function that, given some columns of the dataframe, drops rows of duplicates
def get_duplicate_rows(data):
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
    return data.drop(rows_to_drop, axis=0)

# given columns containing a list of values, convert this column to multiple binary columns; one for each possible
# value
# example:
# original column -
# [a, b, c, d, e]
# [e, f]
# new -
# [0-1] [0-1] [0-1] [0-1] [0-1] [0-1]
def expand_columns(data):
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

    return data

# Drop rows that do not contain certain jurisdiction names
def filter_on_jurisdiction():
    keep_list = ['{Amsterdam," NL"}',
                 '{Amsterdam," NL Zip Codes 2"," Amsterdam"," NL"}',
                 '{Amsterdam}',
                 '{Amsterdam," NL Zip Codes 2"}',
                 np.nan]

    func = {'jurisdiction_names': lambda x: x not in keep_list}
    return pdp.RowDrop(func)

# uniformize missing values by replacing them by NaNs
def uniformize_missing(columns):
    missing_values_strings = ['NaN', '??', '*', 'UNK', '-', '###']
    func = lambda x: np.nan if x in missing_values_strings else x

    return pdp.ApplyByCols(columns, func)

# uniformize boolean values by replacing them by 1 (True) or 0 (False)
def uniformize_boolean(columns):
    true_strings = ['t', 'true', 'yes', 'y', True]
    false_strings = ['f', 'false', 'n', 'no', False]

    func_true = lambda x: 1.0 if x in true_strings else x
    func_false = lambda x: 0.0 if x in false_strings else x

    return pdp.ApplyByCols(columns, func_true) + pdp.ApplyByCols(columns, func_false)

# define a function that unifies the data formats into one format
def unify_datetimes(data):
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
    return data

# convert monetary values to float by stripping the '$' and casting to numeric value
def uniformize_monetary():
    monetary_columns = ['price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee', 'extra_people']

    func = lambda x: float(x[1:].replace(',', '')) if type(x) == str else x

    return pdp.ApplyByCols(monetary_columns, func)

# convert percentage values to float by stripping the '%' and casting to numeric value
def uniformize_percentage():
    percentage_columns = ['host_response_rate']

    func = lambda x: float(x[:-1]) if type(x) == str else x
    return pdp.ApplyByCols(percentage_columns, func)

# drop columns that only contain either 1 possible values or NaNs
def drop_useless():
    useless_column = ['experiences_offered', 'has_availability', 'requires_license', 'is_business_travel_ready']
    return pdp.ColDrop(useless_column)

# build the data cleaning pipeline and return the built pipeline object
def build_pipeline(column_names):
    print("Now building pipeline...")
    n_steps = 9
    # initialize the pipeline
    pipeline = filter_on_jurisdiction()
    print("Step {0} of {1} complete!".format(1, n_steps))
    # get rid of duplicates
    pipeline += pdp.AdHocStage(transform=get_duplicate_rows)
    print("Step {0} of {1} complete!".format(2, n_steps))
    # uniformize missing values
    pipeline += uniformize_missing(column_names)
    print("Step {0} of {1} complete!".format(3, n_steps))
    # uniformize boolean values
    pipeline += uniformize_boolean(column_names)
    print("Step {0} of {1} complete!".format(4, n_steps))
    pipeline += pdp.AdHocStage(transform=unify_datetimes)
    print("Step {0} of {1} complete!".format(5, n_steps))
    pipeline += uniformize_monetary()
    print("Step {0} of {1} complete!".format(6, n_steps))
    pipeline += uniformize_percentage()
    print("Step {0} of {1} complete!".format(7, n_steps))
    pipeline += pdp.AdHocStage(transform=expand_columns)
    print("Step {0} of {1} complete!".format(8, n_steps))
    pipeline += drop_useless()
    print("Step {0} of {1} complete!".format(9, n_steps))

    return pipeline

# clean the dataset located at path_in and store the cleaned data at location path_out
def clean_dataset(path_in, path_out):
    # read in the data from path_in
    data = pd.read_csv(path_in)
    # build the data cleaning pipeline
    pipeline = build_pipeline(data.columns)
    print("Cleaning data with pipeline..")
    # clean data with pipeline
    data_cleaned = pipeline.apply(data, verbose=True)
    # store the data to path_out
    data_cleaned.to_csv(path_out)

def clean_dataset_json(df_json):
    # read in the data from path_in
    data = pd.read_json(df_json)
    # build the data cleaning pipeline
    pipeline = build_pipeline(data.columns)
    print("Cleaning data with pipeline..")
    # clean data with pipeline
    data_cleaned = pipeline.apply(data, verbose=True)
    # return jsonified version of cleaned dataset
    data_cleaned_json = data_cleaned.to_json()

    return data_cleaned_json
