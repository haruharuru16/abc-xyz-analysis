from json import load
import pandas as pd
import numpy as np
import pytest
from abc_functions import abc_classification, load_data, filter_dataset, merge_data
from pandas._testing import assert_frame_equal
from numpy.testing import assert_array_equal


def test_load_data():
    # get data using function
    df_func = load_data('datatest.csv')

    # making datatest
    data = {
        'Product_Code': ['Product_1891', 'Product_1674', 'Product_0287', 'Product_0204', 'Product_1250',
                         'Product_1439', 'Product_0934', 'Product_1107', 'Product_0206', 'Product_2055',
                         'Product_1344', 'Product_1752', 'Product_0841', 'Product_0773', 'Product_1496'],
        'Date': ['01/07/2016', '01/07/2016', '01/07/2016', '01/07/2016', '01/07/2016',
                 '01/07/2016', '01/07/2016', '01/07/2016', '01/07/2016', '01/07/2016',
                 '01/07/2016', '01/07/2016', '01/07/2016', '01/07/2016', '01/07/2016'],
        'Order_Demand': [4, 6, 1200, 105, 2000,
                         2000, 2500, 400, 450, 81,
                         1000, 2, 10, 50, 200]
    }
    df_test = pd.DataFrame(data=data)

    # make sure they are the same
    assert_frame_equal(df_func, df_test)


def test_filter_dataset_1():
    # get data using function
    period = 'Last 3 Months'
    data = pd.read_csv('Product Demand 6 Months.csv')
    df_func = filter_dataset(data, period)

    # get test data
    df_test = pd.read_csv('data_3_months.csv')
    df_test['Date'] = pd.to_datetime(df_test['Date'], dayfirst=True)

    # make sure they are the same
    assert_frame_equal(df_func.reset_index(drop=True),
                       df_test.reset_index(drop=True))


def test_filter_dataset_2():
    # get data using function
    period = '6 Months'
    data = pd.read_csv('Product Demand 6 Months.csv')
    df_func = filter_dataset(data, period)

    # get test data
    df_test = pd.read_csv('Product Demand 6 Months.csv')
    df_test['Date'] = pd.to_datetime(df_test['Date'], dayfirst=True)

    # make sure they are the same
    assert_frame_equal(df_func.reset_index(drop=True),
                       df_test.reset_index(drop=True))


def test_abc_classification():
    # get data using function
    data = load_data('datatest.csv')
    data_abc, abc_class = abc_classification(data, 10, 15)

    # data testing
    df_test = {
        'class': ['A', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']
    }
    df_test = pd.DataFrame(data=df_test)

    # make sure they are the same
    assert_array_equal(data_abc['class'].values,
                       df_test['class'].values, verbose=True)
    pass


# def test_merge_data():
#     # defining data to merge
#     data_1 = {
#         'Product_Code': ['Product_001', 'Product_002', 'Product_003', 'Product_004', 'Product_005'],
#         'class': ['A', 'B', 'C', 'C', 'C']
#     }

#     data_2 = {
#         'Product_Code': ['Product_001', 'Product_002', 'Product_003', 'Product_004', 'Product_005'],
#         'class': ['X', 'Y', 'Y', 'Z', 'Z']
#     }

#     # apply function
#     data_merge = merge_data(data_1, data_2)

#     # data testing
#     df_test = {
#         'class': ['A']
#     }


# def test_abc_xyz_class():
#     pass
