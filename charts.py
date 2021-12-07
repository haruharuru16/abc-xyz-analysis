from threading import Condition
import streamlit as st
from helper import abc_classification, abc_xyz_class, filter_dataset, load_css, load_data, summary_poster, xyz, xyz_classification, xyz_summary
from recommendation import get_recommendation

st.set_page_config(layout='wide')
st.title('ABC-XYZ Classification Dashboard')

# load dataset
dataset = load_data('Product Demand 3 Months - Copy.csv')

# load css
load_css('style.css')

a_input, b_input, duration_input = st.columns([1, 1, 1])
with a_input:
    a_percentage = st.number_input('A Class Percentage')

with b_input:
    b_percentage = st.number_input('B Class Percentage')

with duration_input:
    month_duration = st.selectbox(
        'Classification Period', ('First 2 Months', 'Last 2 Months', '3 Months'))
    filtered_data = filter_dataset(dataset, month_duration)


column1, column2 = st.columns([6.5, 3.5])

with column1:
    st.markdown("**ABC Analysis Result**")

    # Classify data based on ABC classification
    data_abc, abc_class = abc_classification(
        filtered_data, a_percentage, b_percentage)

    # SUMMARY POSTER
    fig = summary_poster(abc_class)
    st.write(fig)

with column2:
    st.markdown("**XYZ Analysis Result**")

    # Classify data based on XYZ Classification plus month duration
    data_xyz, xyz_monthly, xyz_demand = xyz_classification(filtered_data)

    # bar = bar_poster(xyz_monthly, xyz_demand, month_duration)
    bar = xyz_summary(xyz_monthly, xyz_demand, month_duration)
    st.write(bar)

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
