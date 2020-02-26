# this practice streamlit dashboard explores the cleaned airbnb dataset

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gp
import matplotlib.pyplot as plt

@st.cache
def load_data(data_path, geo_data_path):
    data = pd.read_csv(data_path)
    data_geo =  gp.read_file(geo_data_path)
    return data, data_geo

def plot_by_latlon(data):
    st.map(data)

st.write("This interactive dashboard explores the Airbnbs located in Amsterdam.")
# read in the cleaned dataset and the amsterdam geometry data
data_cleaned, data_geo = load_data(data_path='../data/listings_cleaned.csv',
                                   geo_data_path='../data/neighbourhoods.geojson')

show_airbnbs = st.checkbox("Show Airbnbs on map")

if(show_airbnbs):
    plot_by_latlon(data_cleaned[['latitude', 'longitude']].dropna())


