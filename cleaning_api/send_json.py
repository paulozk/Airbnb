# this script sends a JSON-ified pandas dataframe to the flask api
import pandas as pd
import requests

# read the data
data = pd.read_csv('../data/listings.csv')
# jsonify the data
data_json = data.to_json()
# send json file to flask api
response = requests.get("http://localhost:5000/v1/methods/cleaning", json=data_json)
data_cleaned_json = response.json()

# response should be a json-ified version of the cleaned dataset
# convert back to pandas dataframe, then store to disk
data_cleaned = pd.DataFrame(data_cleaned_json)
data_cleaned.to_csv("../data/listings_cleaned.csv")

