import pandas as pd
import pdpipe as pdp
import numpy as np
import sys
import time
from scipy.stats import mode
from sklearn.decomposition import PCA

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

def drop_textual_columns():
    start_time = time.time()
    text_columns = ['name', 'summary', 'listing_url', 'space', 'description', 'neighborhood_overview', 'notes',
                    'transit', 'access',
                    'interaction', 'house_rules', 'picture_url', 'host_url', 'host_name', 'host_picture_url']

    result = pdp.ColDrop(text_columns)

    time_elapsed = time.time() - start_time
    print("drop_textual_columns:", time_elapsed)

    return result

def drop_host_location():
    start_time = time.time()

    result = pdp.ColDrop('host_location')

    time_elapsed = time.time() - start_time
    print("drop_textual_columns:", time_elapsed)

    return result

def drop_geometrical_columns():
    start_time = time.time()

    geo_columns = ['geometry', 'latitude', 'longitude']

    result = pdp.ColDrop(geo_columns)

    time_elapsed = time.time() - start_time
    print("drop_geometrical_columns:", time_elapsed)

    return result

def filter_verifications_and_amenities(data):
    start_time = time.time()

    # check if any of the amenities types or host_verifications types has any useful information or not
    verification_columns = data.columns[data.columns.map(lambda x: "host_verifications" in x)]
    amenities_columns = data.columns[data.columns.map(lambda x: "amenities" in x)]

    # if sum of boolean column values is zero, the column only consists of 0's (and possibly NaNs), so drop the column
    columns_to_delete = []

    # identify useless columns for verification columns
    for col in verification_columns:
        if (data[col].sum() == 0):
            columns_to_delete.append(col)

    # identify useless columns for amenities columns
    for col in amenities_columns:
        if (data[col].sum() == 0):
            columns_to_delete.append(col)

    # drop the useless columns
    data = data.drop(columns_to_delete, axis=1)

    # drop host_id
    data = data.drop(['host_id'], axis=1)

    time_elapsed = time.time() - start_time
    print("filter_verifications_and_amenities:", time_elapsed)

    return data

def convert_calender_updated_to_numeric(data):
    start_time = time.time()

    # replace 'never' with -1, 'today' with 0, 'yesterday' with 1 and 'a week ago' with 7
    data['calendar_updated'] = data['calendar_updated'].dropna().map(lambda x: '-1' if x == 'never' else x)
    data['calendar_updated'] = data['calendar_updated'].dropna().map(lambda x: '0' if x == 'today' else x)
    data['calendar_updated'] = data['calendar_updated'].dropna().map(lambda x: '1' if x == 'yesterday' else x)
    data['calendar_updated'] = data['calendar_updated'].dropna().map(lambda x: '7' if x == 'a week ago' else x)

    # retrieve indices for months and weeks, so they can be multiplied by 7 (weeks) or 30 (months)
    idx_weeks = np.where(data['calendar_updated'].str.contains("weeks").dropna())[0]
    idx_months = np.where(data['calendar_updated'].str.contains("months").dropna())[0]

    # convert to float
    data['calendar_updated'] = data['calendar_updated'].dropna().map(lambda x: x.split()[0]).astype(float)

    # update weeks values
    data['calendar_updated'].iloc[idx_weeks] = data['calendar_updated'].dropna()[idx_weeks] * 7

    # update months values
    data['calendar_updated'][idx_months] = data['calendar_updated'].dropna()[idx_months] * 30

    time_elapsed = time.time() - start_time
    print("convert_calender_updated_to_numeric:", time_elapsed)

    return data

def drop_bed_and_property_type_columns(data):
    start_time = time.time()

    # The columns bed_type and property_type are very sparse, they might not be very discriminative features
    data = data.drop(['bed_type', 'property_type'], axis=1)

    time_elapsed = time.time() - start_time
    print("drop_bed_and_property_type_columns:", time_elapsed)

    return data

def encode_host_response_time_as_ordinal(data):
    start_time = time.time()

    # identify the different values of the column
    idx_nan = np.where(data['host_response_time'].isnull())[0]
    idx_within_an_hour = np.where(data['host_response_time'] == 'within an hour')[0]
    idx_within_a_few_hours = np.where(data['host_response_time'] == 'within a few hours')[0]
    idx_within_a_day = np.where(data['host_response_time'] == 'within a day')[0]
    idx_a_few_days_or_more = np.where(data['host_response_time'] == 'a few days or more')[0]

    # zero-encode the NaN values
    data['host_response_time'][idx_nan] = 0
    data['host_response_time'][idx_within_an_hour] = 1
    data['host_response_time'][idx_within_a_few_hours] = 2
    data['host_response_time'][idx_within_a_day] = 3
    data['host_response_time'][idx_a_few_days_or_more] = 4

    data['host_response_time'] = data['host_response_time'].astype(float)

    time_elapsed = time.time() - start_time
    print("encode_host_response_time_as_ordinal:", time_elapsed)

    return data


def label_encode_host_neighbourhood(data):
    start_time = time.time()

    # host_neighbourhood has too many locations; encode with a number for now
    dict_host_neighbourhood = {neighbourhood: index for index, neighbourhood in
                               enumerate(data['host_neighbourhood'].unique())}

    # apply the dictionary mapping
    data['host_neighbourhood'] = data['host_neighbourhood'].map(dict_host_neighbourhood)

    time_elapsed = time.time() - start_time
    print("label_encode_host_neighbourhood:", time_elapsed)

    return data

def dummy_encode_nominal_columns(data):
    start_time = time.time()

    to_be_dummy_encoded = ['room_type', 'license', 'jurisdiction_names', 'neighbourhood', 'cancellation_policy']

    # dummy encode each column, give NaNs their own category
    for col in to_be_dummy_encoded:


        if (data[col].isnull().sum() > 0):
            dummy_na = True
        else:
            dummy_na = False

        dummy_columns = pd.get_dummies(data[col], prefix=col, dummy_na=dummy_na)
        # drop source column
        data = data.drop(col, axis=1)
        # add columns to data frame with a proper prefix
        data = pd.concat([data, dummy_columns], axis=1)

    time_elapsed = time.time() - start_time
    print("dummy_encode_nominal_columns:", time_elapsed)

    return data

def convert_all_to_numeric(data):
    start_time = time.time()

    # make sure that numeric columns are actually numeric (float)
    for col in data.columns:
        if (data[col].dtype != 'float64' and data[col].dtype != 'object'):
            data[col] = data[col].astype(float)

    time_elapsed = time.time() - start_time
    print("convert_all_to_numeric:", time_elapsed)

    return data

def drop_datetime_columns():
    start_time = time.time()
    time_columns = ['first_review', 'last_review', 'host_since']

    result = pdp.ColDrop(time_columns)

    time_elapsed = time.time() - start_time
    print("drop_datetime_columns:", time_elapsed)

    return result

def drop_columns_with_many_NaNs(data, **kwargs):
    start_time = time.time()

    drop_threshold = kwargs['drop_threshold']
    # check the percentage of NaN values in each column
    n_rows = data.shape[0]
    for col in data.columns:
        n_missing = data[col].isnull().sum()
        # drop columns where the majority of the values are missing
        if ((n_missing / n_rows) > drop_threshold):
            data = data.drop(col, axis=1)

    time_elapsed = time.time() - start_time
    print("drop_columns_with_many_NaNs:", time_elapsed)

    return data

def mean_impute(data):
    start_time = time.time()

    numerical_columns = ['host_response_rate',
     'host_listings_count',
     'host_total_listings_count',
     'host_has_profile_pic',
     'host_identity_verified',
     'is_location_exact',
     'accommodates',
     'bathrooms',
     'bedrooms',
     'beds',
     'price',
     'security_deposit',
     'cleaning_fee',
     'guests_included',
     'extra_people',
     'minimum_nights',
     'maximum_nights',
     'availability_30',
     'availability_60',
     'availability_90',
     'availability_365',
     'number_of_reviews',
     'number_of_reviews_ltm',
     'review_scores_rating',
     'review_scores_accuracy',
     'review_scores_cleanliness',
     'review_scores_checkin',
     'review_scores_communication',
     'review_scores_location',
     'review_scores_value',
     'instant_bookable',
     'require_guest_profile_picture',
     'require_guest_phone_verification',
     'calculated_host_listings_count',
     'calculated_host_listings_count_entire_homes',
     'calculated_host_listings_count_private_rooms',
     'calculated_host_listings_count_shared_rooms',
     'reviews_per_month',
     'calendar_updated']

    for col in numerical_columns:
        if (col in data.columns and data[col].isnull().sum() > 0):
            idx_missing = data[col].isnull()
            data[col][idx_missing] = data[col].dropna().mean()

    time_elapsed = time.time() - start_time
    print("mean_impute:", time_elapsed)

    return data

# mode impute the amenities and verifications binary columns
def mode_impute(data):
    start_time = time.time()

    amenities_and_verifications_columns = list(
        filter(lambda x: ("amenities" in x) or ("host_verifications" in x), data.columns))
    for col in amenities_and_verifications_columns:
        if (data[col].isnull().sum() > 0):
            idx_missing = data[col].isnull()
            data[col][idx_missing] = mode(data[col].dropna())[0][0]

    time_elapsed = time.time() - start_time
    print("mode_impute:", time_elapsed)

    return data


def pca_transform(data):
    start_time = time.time()

    pca = PCA()

    # fit an pca object on the data
    pca.fit(data.drop('id', axis=1))
    # take first 10 principle components
    data_listings_pca = pd.DataFrame(pca.transform(data.drop('id', axis=1))[:, :10])
    # add 'id' column to pca version of listings data
    data_listings_pca = pd.concat([data_listings_pca, data['id']], axis=1)

    time_elapsed = time.time() - start_time
    print("pca_transform:", time_elapsed)

    return data_listings_pca

def drop_missing_ids(data):
    start_time = time.time()

    idx_missing = np.where(data.id.isnull())[0]
    data = data.drop(idx_missing, axis=0)

    time_elapsed = time.time() - start_time
    print("drop_missing_ids:", time_elapsed)

    return data


def preprocess_calendar_data(data):
    data = data.rename(columns={'listing_id': 'id'})
    # drop date column
    data = data.drop('date', axis=1)
    # convert boolean (f-t) to 0s and 1s
    data['available'][data['available'] == 'f'] = 0
    data['available'][data['available'] == 't'] = 1
    # convert monetary columns to numeric
    #data['price'] = data['price'].map(lambda x: x[1:]
    data['price'] = data['price'].str.replace('$', '')
    data['price'] = data['price'].str.replace(',', '')
    #data['adjusted_price'] = data['adjusted_price'].map(lambda x: x[1:])
    data['adjusted_price'] = data['adjusted_price'].str.replace('$', '')
    data['adjusted_price'] = data['adjusted_price'].str.replace(',', '')

    # convert to numeric
    data = data.astype(float)

    print(data.head())

    return data

def merge_with_calendar(data, **kwargs):
    idx_missing = np.where(data.id.isnull())[0]
    data = data.drop(idx_missing, axis=0)

    data_calendar = kwargs['data_calendar']

    # preprocess the calendar data
    data_calendar = preprocess_calendar_data(data_calendar)

    # merge the two datasets on the 'id' column
    data_listings_and_calendar = data_calendar.merge(data, on='id')

    return data_listings_and_calendar



# build the data preprocessing pipeline and return the built pipeline object
def build_pipeline(data_calendar):

    # PREPROCESSING STAGES

    print("Now building pipeline...")
    # initialize the pipeline by dropping textual columns
    pipeline = drop_textual_columns()
    # drop the host location column
    pipeline += drop_host_location()
    # drop geometrical columns
    pipeline += drop_geometrical_columns()
    # drop useless 'amenities' and 'host_verifications' columns
    pipeline += pdp.AdHocStage(transform=filter_verifications_and_amenities)
    # convert the column 'calender_updated' from string to numeric
    pipeline += pdp.AdHocStage(transform=convert_calender_updated_to_numeric)
    # drop bed_type and property_type columns
    pipeline += pdp.AdHocStage(transform=drop_bed_and_property_type_columns)
    # encode 'host_response_time' as an ordinal variable (0-4)
    pipeline += pdp.AdHocStage(transform=encode_host_response_time_as_ordinal)
    # label encode the host neighbourhood column
    pipeline += pdp.AdHocStage(transform=label_encode_host_neighbourhood)
    # dummy encode some nominal columns
    pipeline += pdp.AdHocStage(transform=dummy_encode_nominal_columns)
    # convert column types to float
    pipeline += pdp.AdHocStage(transform=convert_all_to_numeric)
    # drop columns related to date/time
    pipeline += drop_datetime_columns()
    # set threshold for dropping columns; columns with proportion of MaNs > threshold will be dropped
    drop_threshold = 0.5
    kwargs = {'drop_threshold': drop_threshold}
    pipeline += AdHogStageArg(transform=drop_columns_with_many_NaNs, **kwargs)
    # mean impute remaining continuous columns
    pipeline += pdp.AdHocStage(transform=mean_impute)
    # mode impute remaining binary columns
    pipeline += pdp.AdHocStage(transform=mode_impute)

    # MERGE STAGES

    # pca transform the data
    pipeline += pdp.AdHocStage(transform=pca_transform)
    # drop rows with missing IDs
    pipeline += pdp.AdHocStage(transform=drop_missing_ids)
    # merge data with calendar data
    kwargs = {'data_calendar': data_calendar}
    pipeline += AdHogStageArg(transform=merge_with_calendar, **kwargs)

    return pipeline


# clean the dataset located at path_in and store the cleaned data at location path_out
def preprocess_dataset(path_in, path_out, path_calendar):
    # read in the data from path_in
    data = pd.read_csv(path_in)
    print("Starting data shape:", data.shape)
    data_calendar = pd.read_csv(path_calendar)
    # build the data cleaning pipeline
    pipeline = build_pipeline(data_calendar)
    print("Cleaning data with pipeline..")
    # clean data with pipeline
    data_preprocessed = pipeline.apply(data, verbose=False)
    print("preprocessed data shape:", data_preprocessed.shape)
    # store the data to path_out
    data_preprocessed.to_csv(path_out, index=None)


if(__name__ == '__main__'):
    try:
        _, path_in, path_out, path_calendar = sys.argv
    except:
        print("Please supply input and output path!")
        sys.exit()

    preprocess_dataset(path_in, path_out, path_calendar)