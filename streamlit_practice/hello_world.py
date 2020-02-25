import streamlit as st
import pandas as pd
import numpy as np

select_box = st.sidebar.selectbox('Choose one if you dare...',
                                  ('None', 'Trick', 'Treat')
                                 )
txt = "Hello :-)"
if(select_box == 'Trick'):
    txt = 'Boo.'
elif(select_box == 'Treat'):
    txt = 'Congrats! You get nothing.'

st.write(txt)

@st.cache  # 👈 This function will be cached
def my_slow_function(arg1):
    # Do something really slow in here!
    sum = 0
    for i in range(arg1 ** 2):
        sum += i
    return sum

st.write("Large sum result:", my_slow_function(5000))

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)