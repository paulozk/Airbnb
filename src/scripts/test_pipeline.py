from Airbnb.src.scripts.data_cleaning import *
import pandas as pd
import time

# read in the data
data = pd.read_csv("../data/listings.csv")
# build the data cleaning pipeline
pipeline = build_pipeline(data.columns)
print("Cleaning data with pipeline..")
# time the data cleaning pipeline
start = time.time()
# clean data with pipeline
data_cleaned = pipeline.apply(data, verbose=True)
end = time.time()
print(data_cleaned)
print("Time elapsed: {0} seconds".format(end - start))

data_cleaned.to_csv("../data/listings_cleaned.csv")
