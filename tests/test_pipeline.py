from Airbnb.cleaning.data_cleaning import clean_dataset
import time

# time the data cleaning pipeline
start = time.time()
# clean dataset
clean_dataset("../data/listings.csv", "../data/listings_cleaned.csv")
end = time.time()
print("Time elapsed: {0} seconds".format(end - start))


