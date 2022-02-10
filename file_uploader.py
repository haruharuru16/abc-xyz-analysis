import pandas as pd
import streamlit as st

# upload file function


def file_uploader(filename, filetype):
    # Function to upload file from user
    # check filetype and not None
    if filetype == 'EXCEL' and (filename is not None):
        dataset = pd.read_excel(filename)

    elif filetype == 'CSV' and (filename is not None):
        dataset = pd.read_csv(filename)

    else:
        dataset = pd.read_csv('Product Demand 6 Months.csv')

    return dataset


def df_downloader(filename):
    # function to download analysis result
    return filename.to_csv().encode('utf-8')
