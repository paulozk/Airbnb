from flask import render_template, request, make_response
from app import app
from cleaning.data_cleaning import *
from preprocessing.data_preprocessing import preprocess_dataset_csv
import geopandas as gpd
import json

def cleaning(data, data_geo):
    # read in the data from path_in
    data = pd.read_csv(data)
    # read in data from data_geo json file
    data_geo = gpd.read_file(data_geo)

    # send json file to cleaning package for cleaning and return json-ified clean dataset
    dataset_cleaned = clean_dataset_csv(data, data_geo)

    # return clean dataset
    return dataset_cleaned


def preprocessing(data_listings, data_calendar):
    # read in the data from path_in
    data_listings = pd.read_csv(data_listings)
    # read in data from data_geo json file
    data_calendar = pd.read_csv(data_calendar)
    # preprocess data with the preprocessing package
    dataset_preprocessed = preprocess_dataset_csv(data_listings, data_calendar)

    # return preprocessed dataset
    return dataset_preprocessed



@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/v1/methods/cleaning', methods=['POST'])
def clean_data():
    # extract data from request
    json_str = request.get_json()
    json_dict = json.loads(json_str)

    data = json_dict['listings']
    data_geo = json_dict['geo']

    # read datasets from JSON
    data = pd.read_json(data)
    data_geo = gpd.GeoDataFrame.from_features(data_geo)

    # send json file to cleaning package for cleaning and return json-ified clean dataset
    dataset_cleaned = clean_dataset_csv(data, data_geo)
    # JSON=ify cleaned data set and return it
    print(dataset_cleaned.head())
    data_cleaned_json = dataset_cleaned.to_json()


    # return clean dataset .csv file to requester
    #response = make_response(dataset_cleaned.to_csv())
    #response.headers["Content-Disposition"] = "attachment; filename=cleaned_data.csv"
    #response.headers["Content-Type"] = "text/csv"

    return data_cleaned_json


@app.route('/v1/methods/preprocessing', methods=['GET'])
def preprocess_data():
    if (request.method == 'POST'):
        # try to extract two csv files

        data_listings = request.files['listings']
        data_calendar = request.files['calendar']

        # preprocess the dataset
        dataset_preprocessed = preprocessing(data_listings, data_calendar)

        # return clean dataset .csv file to requester
        response = make_response(dataset_preprocessed.iloc[:10].to_csv())
        response.headers["Content-Disposition"] = "attachment; filename=listings_preprocessed.csv"
        response.headers["Content-Type"] = "text/csv"

        return response


        #except:
        #    return "<h1>Please send the listings and calendar data in .csv format!<h1>"



@app.route('/v1/methods/all', methods=['POST'])
def clean_and_preprocess_data():
    if (request.method == 'POST'):
        # try to extract two csv files

        data = request.files['listings']
        data_geo = request.files['geo']
        data_calendar = request.files['calendar']

        # clean the data
        data_cleaned = cleaning(data, data_geo)
        # read in data from data_geo json file
        data_calendar = pd.read_csv(data_calendar)
        # preprocess data with the preprocessing package
        data_preprocessed = preprocess_dataset_csv(data_cleaned, data_calendar)

        # return clean dataset .csv file to requester
        response = make_response(data_preprocessed.iloc[:100].to_csv())
        response.headers["Content-Disposition"] = "attachment; filename=listings_preprocessed.csv"
        response.headers["Content-Type"] = "text/csv"

        return response


