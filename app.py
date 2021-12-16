from threading import Condition
import streamlit as st
from abc_functions import abc_classification, abc_xyz_class, filter_dataset, load_css, load_data, summary_poster
from xyz_functions import xyz_classifier, xyz_poster
from recommendation_function import get_recommendation

st.set_page_config(layout='wide')
st.title('ABC-XYZ Classification Dashboard')

# load dataset
dataset = load_data('Product Demand 6 Months.csv')

# load css
load_css('style.css')

duration_input, a_input, b_input, x_input, y_input = st.columns([
                                                                1, 1, 1, 1, 1])
with a_input:
    a_percentage = st.number_input('A Class Percentage')

with b_input:
    b_percentage = st.number_input('B Class Percentage')

with duration_input:
    month_duration = st.selectbox(
        'Classification Period', ('6 Months', 'Last 3 Months'))
    filtered_data = filter_dataset(dataset, month_duration)

with x_input:
    x_percentage = st.number_input('X Class Percentage')

with y_input:
    y_percentage = st.number_input('Y Class Percentage')


# ABC Classification Summary Dashboard
st.markdown("**ABC Analysis Result**")

# Classify data based on ABC classification
data_abc, abc_class = abc_classification(
    filtered_data, a_percentage, b_percentage)

# ABC Classification Summary Poster
fig = summary_poster(abc_class)
st.write(fig)

# XYZ Classification Summary Dashboard
st.markdown("**XYZ Analysis Result**")

# Classify data based on XYZ Classification plus month duration
data_xyz, xyz_monthly, xyz_demand = xyz_classifier(
    filtered_data, x_percentage, y_percentage)

bar = xyz_poster(xyz_monthly, xyz_demand, month_duration)
st.write(bar)

# ABC-XYZ Classificatio Summary Table and Recommendation
coltab1, coltab2 = st.columns(2)

with coltab1:
    st.markdown("**ABC-XYZ Analysis Classification Result**")

    # combining the classification results
    abc_xyz_data = abc_xyz_class(abc_class, xyz_demand)
    st.write(abc_xyz_data)

with coltab2:
    st.markdown("**ABC Analysis Reevaluation Recommendation**")

    # getting recommendation
    data_1 = get_recommendation(dataset, a_percentage, b_percentage)
    st.write(data_1)
