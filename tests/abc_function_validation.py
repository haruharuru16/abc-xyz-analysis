from typing import Callable, List, NoReturn
import pandas as pd
import numpy as np
import pytest
from abc_functions import abc_classification, abc_xyz_class, filter_dataset, load_data, merge_data
from numpy.testing import assert_array_equal


def get_datatest_2() -> pd.DataFrame:
    """
    Second datatest
    """
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

    data_test = pd.DataFrame(data=data)

    return data_test


def get_datatest_6_months() -> pd.DataFrame:
    """
    Data 6 months
    """
    dataset = pd.read_csv('Product Demand 6 Months.csv')
    dataset['Date'] = pd.to_datetime(dataset['Date'], dayfirst=True)

    return dataset


def get_datatest_3_months() -> pd.DataFrame:
    """
    Data 3 months
    """
    dataset = pd.read_csv('data_3_months.csv')
    dataset['Date'] = pd.to_datetime(dataset['Date'], dayfirst=True)

    return dataset

####################### Testing the Load Function #######################


@pytest.fixture
def datatest_load() -> pd.DataFrame:
    """
    Get the second data test
    """
    datatest = get_datatest_2()

    return datatest


def test_load_data(datatest_load: Callable):
    """
    Test if the data contained from the actual dataset is the same
    with the data built from load_data function
    """

    # get the test data
    df_func = load_data('datatest.csv')

    # testing whether they are the same
    assert datatest_load.iloc[3, 2] == df_func.iloc[3, 2]

####################### Testing the Filter Dataset 1 #######################


@pytest.fixture
def datatest_3_months_load() -> pd.DataFrame:
    """
    Get 3 months dataset to filter
    """
    datatest = get_datatest_3_months()

    return datatest


def test_filter_dataset_3_months(datatest_3_months_load: Callable):
    """
    Test whether the filter function for 3 months is working
    """
    # get the test data
    data = pd.read_csv('Product Demand 6 Months.csv')

    # use the filter function
    data_filter_3 = filter_dataset(data, 'Last 3 Months')

    # validate
    assert data_filter_3.iloc[5, 1] == datatest_3_months_load.iloc[5, 1]


####################### Testing the Filter Dataset 2 #######################

@pytest.fixture
def datatest_6_months_load() -> pd.DataFrame:
    """
    Get 6 months dataset to filter
    """
    datatest = get_datatest_6_months()

    return datatest


def test_filter_dataset_6_months(datatest_6_months_load: Callable):
    """
    Test whether the filter function for 6 months is working
    """
    # get the test data
    data = pd.read_csv('Product Demand 6 Months.csv')

    # use the filter function
    data_6 = filter_dataset(data, '6 Months')

    # validate
    assert data_6.iloc[5, 1] == datatest_6_months_load.iloc[5, 1]


####################### Testing the ABC Classification #######################

def test_abc_classification(datatest_load: Callable):
    """
    Testing the result of ABC Classification
    """
    # using the abc function
    data_abc, abc_class = abc_classification(datatest_load, 10, 15)

    # expected classes
    data_class = {
        'class': ['A', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']
    }
    df_test = pd.DataFrame(data=data_class)

    assert_array_equal(data_abc['class'].values,
                       df_test['class'].values, verbose=True)


####################### Testing the Merge Function #######################

def test_merge_data():
    """
    Testing the merge function
    First data has 2 columns, second data has 2 columns
    When merged using a similar column, it should return a (n, 3) shaped dataframe
    """

    # defining data to merge
    data_1 = {
        'Product_Code': ['Product_001', 'Product_002', 'Product_003', 'Product_004', 'Product_005'],
        'class': ['A', 'B', 'C', 'C', 'C']
    }
    data_1 = pd.DataFrame(data=data_1)

    data_2 = {
        'Product_Code': ['Product_001', 'Product_002', 'Product_003', 'Product_004', 'Product_005'],
        'class': ['X', 'Y', 'Y', 'Z', 'Z']
    }
    data_2 = pd.DataFrame(data=data_2)

    # apply function
    data_merge = merge_data(data_1, data_2)

    # validate
    assert data_merge.shape[1] == 3

####################### Testing the ABC-XYZ Function #######################


def test_abc_xyz():
    """
    Testing the abc xyz function
    """

    # defining data to merge
    data_1 = {
        'Product_Code': ['Product_001', 'Product_002', 'Product_003', 'Product_004', 'Product_005'],
        'Order_Demand': [100, 50, 20, 15, 15],
        'class': ['A', 'B', 'C', 'C', 'C'],
    }
    data_1 = pd.DataFrame(data=data_1)

    data_2 = {
        'Product_Code': ['Product_001', 'Product_002', 'Product_003', 'Product_004', 'Product_005'],
        'Order_Demand': [100, 50, 20, 15, 15],
        'class': ['X', 'Y', 'Y', 'Z', 'Z']
    }
    data_2 = pd.DataFrame(data=data_2)

    # apply function
    data_merge = abc_xyz_class(data_1, data_2)

    # validate
    assert_array_equal(data_merge['abc_class'].values, data_1['class'].values)
