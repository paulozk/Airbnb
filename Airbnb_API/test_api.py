# this script sends JSON files to the flask api for cleaning and preprocessing the airbnb dataset
import pandas as pd
import requests
from requests_toolbelt import MultipartEncoder
import json

# read the data
data_listings = pd.read_csv('../data/listings.csv', nrows=1000)
print('Listings data loaded!')
data_calendar = pd.read_csv('../data/calendar.csv', nrows=1000)
print('Calendar data loaded!')

# jsonify the dataframes
listings_json = data_listings.to_json()
calendar_json = data_calendar.to_json()
with open('../data/neighbourhoods.geojson') as f:
    geojson = json.load(f)

files = {}
files['listings'] = listings_json
files['calendar'] = calendar_json
files['geo'] = geojson

all_json = json.dumps(files)

print('JSON loaded!')

# send json file to flask api
response = requests.post("http://127.0.0.1:5000/v1/methods/cleaning", json=all_json)
data_cleaned_json = response.json()

# response should be a json-ified version of the cleaned dataset
# convert back to pandas dataframe, then store to disk
data_cleaned = pd.DataFrame(data_cleaned_json)
data_cleaned.to_csv("../data/listings_cleaned.csv")

