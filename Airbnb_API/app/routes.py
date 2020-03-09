from flask import render_template, request, make_response
from app import app
from cleaning.data_cleaning import *
from preprocessing.data_preprocessing import preprocess_dataset
import geopandas as gpd

@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/v1/methods/cleaning/download', methods=['POST'])
def clean_dataset():
    if(request.method == 'POST'):
        data = request.files['file']
        data_geo = request.files['file2']



        if(data and data_geo):
            # read in the data from path_in
            data = pd.read_csv(data)
            # read in data from data_geo json file
            data_geo = gpd.read_file(data_geo)

            # send json file to cleaning package for cleaning and return json-ified clean dataset
            cleaned_dataset = clean_dataset_csv(data, data_geo)

            # return clean dataset .csv file to requester
            response = make_response(cleaned_dataset.to_csv())
            response.headers["Content-Disposition"] = "attachment; filename=cleaned_data.csv"
            response.headers["Content-Type"] = "text/csv"

            return response


        return "<h1>Please send two proper files!<h1>"

