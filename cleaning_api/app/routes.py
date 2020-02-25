from flask import render_template, request
from app import app
from cleaning.data_cleaning import clean_dataset_json

@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/v1/methods/cleaning', methods=['GET'])
def clean_dataset():
    if(request.method == 'GET'):
        # get json-ified pandas dataset that needs cleaning
        content = request.get_json()
        # send json file to cleaning package for cleaning and return json-ified clean dataset
        cleaned_dataset_json = clean_dataset_json(content)

        # return json-ified clean dataset to user
        return cleaned_dataset_json

