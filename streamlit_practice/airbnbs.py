# this practice streamlit dashboard explores the cleaned airbnb dataset

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import json

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



st.write("This interactive dashboard explores the Airbnbs located in Amsterdam.")

# read in the cleaned dataset and the amsterdam geometry data
data_cleaned, data_geo = load_data(data_path='../data/listings_cleaned.csv',
                                   geo_data_path='../data/neighbourhoods.geojson')

# load the json geodata
json_str = load_json('../data/neighbourhoods.geojson')

show_airbnbs = st.checkbox("Show Airbnbs on map")
show_neighbourhoods = st.checkbox("Show neighborhood statistics")

if(show_airbnbs):
    plot_by_latlon(data_cleaned[['latitude', 'longitude']].dropna())

if(show_neighbourhoods):
    neighborhood_column = st.selectbox("Possible columns", ("price", "weekly_price", "monthly_price"))
    fig = plot_by_neighborhood(data_cleaned, neighborhood_column, json_str)






