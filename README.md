# Amsterdam Airbnbs 
This repository contains tools for exploring and processing Airbnb data gathered from Airbnbs in Amsterdam from AirBnB.com. Processing includes:
* Data Cleaning: Dealing with inconsistencies in the data, such as non-uniform markers for missing values
  * Usage:
    * Python:
    ```python
      from cleaning.data_cleaning import clean_dataset
      clean_dataset(PATH_TO_DATA, DESTINATION_PATH)
     ```
    * Flask:
      1: Navigate to cleaning_api folder

      2:
      ```python
      flask run
      ```
      3: 
      ```python
      requests.get("http://localhost:YOUR_PORT/v1/methods/cleaning", json=YOUR_DATASET_AS_JSON)
      ```
      The response will contain the JSON version of the cleaned dataset
  

* Visualization: Visualize insights into the data (per district) and offer this via an interactive dashboard (Flask + Dash)
* Data Preprocessing: Transforming the data to make it machine-readable, so that it can be fed to a machine learning model

# Installation
```cmd
pip install -i https://test.pypi.org/simple/ Airbnb_Processing==0.0.2
```
