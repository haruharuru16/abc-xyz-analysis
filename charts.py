from threading import Condition
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from helper import abc_classification, bar_poster, filter_dataset, load_css, load_data, summary_poster, xyz, xyz_classification

st.set_page_config(layout='wide')
# st.title('ABC-XYZ Classification Dashboard')

# load dataset
dataset = load_data('Product Demand 3 Months - Copy.csv')

# load css
load_css('style.css')

a_input, b_input, xyz_input = st.columns([1, 1, 1])
with a_input:
    a_percentage = st.number_input('A Class Percentage')

with b_input:
    b_percentage = st.number_input('B Class Percentage')

with xyz_input:
    month_duration = st.selectbox(
        'XYZ Classification Period', ('First 2 Months', 'Last 2 Months', '3 Months'))


column1, column2 = st.columns([6.5, 3.5])

with column1:
    st.markdown("**ABC Analysis For Latest 3 Months Data**")

    # Classify data based on ABC classification
    abc_class = abc_classification(dataset, a_percentage, b_percentage)

    # SUMMARY POSTER
    fig = summary_poster(abc_class)
    st.write(fig)

with column2:
    st.markdown("**XYZ Analysis For Latest 3 Months Data**")

    # Classify data based on XYZ Classification plus month duration
    filtered_data = filter_dataset(dataset, month_duration)
    xyz_monthly, xyz_demand = xyz_classification(filtered_data)

    bar = bar_poster(xyz_monthly, xyz_demand, month_duration)
    st.write(bar)

# coltab1, coltab2 = st.columns(2)

# with coltab1:
#     st.write(abc_class)

# with coltab2:
#     st.write(xyz_monthly)
#     st.write(xyz_demand)
