# this practice streamlit dashboard explores the cleaned airbnb dataset

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import json
import requests

@st.cache
def load_data(data_path, geo_data_path):
    data = pd.read_csv(data_path)
    data_geo =  gpd.read_file(geo_data_path)
    return data, data_geo

@st.cache
def load_json(path):
    with open(path) as f:
        json_str = json.load(f)
    return json_str

#@st.cache(suppress_st_warning=True)
def plot_by_latlon(data):
    st.map(data)


def group_and_compute(data, column_name):
    value_by_neighborhood = data.groupby('neighbourhood')[column_name].mean()[1:]
    data_result = pd.DataFrame(value_by_neighborhood)
    data_result = data_result.reset_index()

    return data_result

@st.cache
def connect_to_API(data):
    # make a call to the data cleaning API to clean the data
    # jsonify the data
    st.write('Now JSON-ifying the dataset..')
    data_json = data.to_json()
    st.write('JSONification complete!')
    # send json file to flask api
    st.write("Now cleaning the data with the data cleaning API. This should take up to 2 minutes...")
    response = requests.get("http://localhost:5000/v1/methods/cleaning", json=data_json)
    st.write("Data cleaning successful!")
    data_cleaned_json = response.json()
    # response should be a json-ified version of the cleaned dataset
    # convert back to pandas dataframe and return
    data_cleaned = pd.DataFrame(data_cleaned_json)

    return data_cleaned



def plot_by_neighborhood(df, column_name, json_str):
    df = group_and_compute(df, column_name)
    fig = px.choropleth(df,
                        geojson=json_str,
                        locations='neighbourhood',
                        color_continuous_scale="hot",
                        color=column_name,
                        labels={'column_name': 'price'},
                        featureidkey='properties.neighbourhood',
                        title=("Average AirBNBs {0} per Amsterdam neighborhood".format(column_name)),
                        )

    fig.update_layout(height=500, margin={"r": 0, "t": 50, "l": 0, "b": 0})
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig)
    #fig.show()


x = st.slider('number')
st.write(x)
st.write("This interactive dashboard explores the Airbnbs located in Amsterdam.")

show_airbnbs = st.checkbox("Show Airbnbs on map")
show_neighbourhoods = st.checkbox("Show neighborhood statistics")

# read in the cleaned dataset and the amsterdam geometry data
data_cleaned, data_geo = load_data(data_path='../data/listings_cleaned.csv',
                                   geo_data_path='../data/neighbourhoods.geojson')

#data_cleaned = connect_to_API(data)

# load the json geodata
json_str = load_json('../data/neighbourhoods.geojson')


if(show_airbnbs):
    # take random sample for more clear and quick plotting
    plot_by_latlon(data_cleaned[['latitude', 'longitude']].dropna().sample(n=1000, replace=False))

if(show_neighbourhoods):
    neighborhood_column = st.selectbox("Possible columns", ("price", "weekly_price", "monthly_price", 'host_response_rate'))
    fig = plot_by_neighborhood(data_cleaned, neighborhood_column, json_str)






