from datetime import datetime
import streamlit as st
from abc_functions import abc_classification, abc_xyz_class, filter_dataset, load_data, summary_poster
from file_uploader import file_uploader
from xyz_functions import xyz_classifier, xyz_poster
from recommendation_function import get_xyz_recommendation
from fpdf import FPDF

st.set_page_config(layout='wide')
st.title('ABC-XYZ Classification Dashboard')


# load dataset
# dataset = load_data('Product Demand 6 Months.csv')

#========== Load Dataset ==========#
st.markdown("**Input Data**")

filetype, filename = st.columns([1, 1])
with filetype:
    file_type = st.selectbox('Choose a file type', ('EXCEL', 'CSV'))

with filename:
    file_upload = st.file_uploader('Choose a file')
    dataset = file_uploader(file_upload, file_type)


#========== End of the load data part ==========#

st.markdown('---')

#========== ABC Classification Summary Dashboard ==========#
st.markdown("**ABC Analysis Classification**")

# input a and b percentage
duration_input, a_input, b_input = st.columns([1, 1, 1])
with a_input:
    a_percentage = st.number_input('A Class Percentage')

with b_input:
    b_percentage = st.number_input('B Class Percentage')

with duration_input:
    month_duration = st.selectbox(
        'ABC Classification Period', ('6 Months', 'Last 3 Months'))
    filtered_data = filter_dataset(dataset, month_duration)


# Classify data based on ABC classification
data_abc, abc_class = abc_classification(
    filtered_data, a_percentage, b_percentage)

# ABC Classification Summary Poster
fig = summary_poster(abc_class)
st.write(fig)

#========== End of the first part ==========#

st.markdown("---")

#========== XYZ Classification Summary Dashboard ==========#
st.markdown("**XYZ Analysis Classification**")

xyz_period, x_input, y_input = st.columns([1, 1, 1])

with xyz_period:
    xyz_month_period = st.selectbox(
        'XYZ Classification Period', ('6 Months', 'Last 3 Months'))
    xyz_filtered_data = filter_dataset(dataset, xyz_month_period)

with x_input:
    x_percentage = st.number_input('X Class Percentage')

with y_input:
    y_percentage = st.number_input('Y Class Percentage')

# Classify data based on XYZ Classification plus month duration
data_xyz, xyz_monthly, xyz_demand = xyz_classifier(
    xyz_filtered_data, x_percentage, y_percentage)

bar = xyz_poster(xyz_monthly, xyz_demand, xyz_month_period)
st.write(bar)

#========== End of the second part ==========#

st.markdown("---")

#========== ABC-XYZ Classificatio Summary Table and Recommendation ==========#
coltab1, coltab2 = st.columns(2)

with coltab1:
    st.markdown("**ABC-XYZ Analysis Classification Result**")

    # combining the classification results
    abc_xyz_data = abc_xyz_class(abc_class, xyz_demand)
    st.write(abc_xyz_data)

with coltab2:
    st.markdown("**Analysis Reevaluation Recommendation Based on Seasonality**")

    # getting recommendation
    data_calc, data_rec = get_xyz_recommendation(xyz_filtered_data)
    st.write(data_rec)

#========== End of the third part ==========#

# st.markdown("---")

#========== Algortihm ABC, XYZ Analysis, and XYZ Recommendation Reevaluation Logic Sample ==========#

col1, col2 = st.columns(2)

with col1:
    # ABC Classification
    # st.markdown("**ABC Classification Analysis Logic**")
    data_abc_copy = data_abc.copy()
    data_abc_copy['index'] = data_abc_copy['rank']
    data_abc_copy.set_index('index', inplace=True)
    # st.write(data_abc_copy)

    # Recommendation
    # st.markdown("**XYZ Reevaluation Recommendation Analysis Logic**")
    # st.write(data_calc)

with col2:
    # XYZ Classification
    # st.markdown("**XYZ Classification Analysis Logic**")
    data_xyz_copy = data_xyz.copy()
    data_xyz_copy['index'] = data_xyz_copy['rank']
    data_xyz_copy.set_index('index', inplace=True)
    # st.write(data_xyz_copy)

#========== End of the fourth part ==========#

st.markdown('---')

#========== Download PDF Report ==========#
st.markdown('**Download Report**')

# save image
fig.write_image('images/fig1.png')
bar.write_image('images/bar1.png')


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297

    def header(self) -> None:
        self.image('assets/header-cropped.png', 0, 0, self.WIDTH)
        self.image('assets/logo.png', 10, 8, 60)
        self.set_font('Arial', 'B', 11)
        self.cell(self.WIDTH - 80)
        self.cell(60, 6, 'ABCXYZ Analysis Report', 0, 0, 'R')
        self.cell(0, 15, str(datetime.today().strftime("%d/%m/%Y")), 15, 5, 'R')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')


pdf = PDF()

# Add ABC Classification Report
pdf.add_page()
pdf.set_font('Arial', '', 12)
pdf.set_xy(10.0, 45)
pdf.cell(w=75.0, h=5.0, align='L', txt='ABC Analysis Classification Report')
pdf.ln(10)
pdf.image('images/fig1.png', w=190, h=100)

# Add XYZ Classification Report
pdf.ln(10)
pdf.cell(w=75.0, h=5.0, align='L', txt='XYZ Analysis Classification Report')
pdf.ln(10)
pdf.image('images/bar1.png', w=190, h=100)

st.download_button(
    'Download Report',
    data=pdf.output(dest='S').encode('latin-1'),
    file_name='ABCXYZ Analysis Report.pdf',
)
