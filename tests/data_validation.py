from typing import Callable, List, NoReturn
import pandas as pd
import numpy as np
import pytest
from abc_functions import abc_classification, abc_xyz_class, filter_dataset, load_data, merge_data
from numpy.testing import assert_array_equal


def get_datatest() -> pd.DataFrame:
    """
    Compile dataframe testing to feed into the tests
    """

    data = {
        'Product_Code': ['Product_0934', 'Product_0934', 'Product_0934',
                         'Product_1250', 'Product_1250', 'Product_1250',
                         'Product_1439', 'Product_1439', 'Product_1439',
                         'Product_0287', 'Product_0287', 'Product_0287',
                         'Product_1344', 'Product_1344', 'Product_1344',
                         'Product_0206', 'Product_0206', 'Product_0206',
                         'Product_1107', 'Product_1107', 'Product_1107',
                         'Product_1496', 'Product_1496', 'Product_1496',
                         'Product_0204', 'Product_0204', 'Product_0204',
                         'Product_2055', 'Product_2055', 'Product_2055',
                         'Product_0773', 'Product_0773', 'Product_0773',
                         'Product_0841', 'Product_0841', 'Product_0841',
                         'Product_1674', 'Product_1674', 'Product_1674',
                         'Product_1891', 'Product_1891', 'Product_1891',
                         'Product_1752', 'Product_1752', 'Product_1752'],

        'Date': ['01/07/2016', '01/08/2016', '01/09/2016',
                 '01/07/2016', '01/08/2016', '01/09/2016',
                 '01/07/2016', '01/08/2016', '01/09/2016',
                 '01/07/2016', '01/08/2016', '01/09/2016',
                 '01/07/2016', '01/08/2016', '01/09/2016',
                 '01/07/2016', '01/08/2016', '01/09/2016',
                 '01/07/2016', '01/08/2016', '01/09/2016',
                 '01/07/2016', '01/08/2016', '01/09/2016',
                 '01/07/2016', '01/08/2016', '01/09/2016',
                 '01/07/2016', '01/08/2016', '01/09/2016',
                 '01/07/2016', '01/08/2016', '01/09/2016',
                 '01/07/2016', '01/08/2016', '01/09/2016',
                 '01/07/2016', '01/08/2016', '01/09/2016',
                 '01/07/2016', '01/08/2016', '01/09/2016',
                 '01/07/2016', '01/08/2016', '01/09/2016'],

        'Order_Demand': [250, 1500, 50,
                         2000, 1000, 1200,
                         6, 0, 10,
                         1000, 3800, 400,
                         450, 2, 12,
                         2, 1, 1,
                         400, 10, 10000,
                         105, 1000, 30,
                         81, 90, 50,
                         200, 2300, 100,
                         50, 32, 44,
                         1200, 5000, 3000,
                         10, 1, 0,
                         2000, 490, 1800,
                         4, 4, 3]
    }

    datatest = pd.DataFrame(data=data)

    return datatest


####################### Testing Duplicates in DataFrame #######################


@pytest.fixture
def duplicated_data() -> pd.DataFrame:
    """
    Get the duplicated rows of the dataset
    """
    datatest = get_datatest()
    datatest_duplicated_df = datatest.duplicated().sum()

    return datatest_duplicated_df


def test_duplicate_in_df(duplicated_data: Callable):
    """
    Test if the duplicated dataframe is empty -> no duplicated rows
    """
    assert duplicated_data == 0


####################### Testing the Data Types in DataFrame #######################

@pytest.fixture
def datatest_data_types() -> dict[str, np.dtype]:
    """
    Get the data types from datatest
    """

    datatest = get_datatest()
    datatest_data_types = datatest.dtypes.to_dict()

    return datatest_data_types


def test_datatest_data_types(datatest_data_types: Callable) -> NoReturn:
    """
    Test the data types of the database and data types of
    the transformed dataframe
    """

    database_schema_data_types = {
        'Product_Code': np.dtype('O'),
        'Date': np.dtype('O'),
        'Order_Demand': np.dtype('int64')
    }

    assert database_schema_data_types == datatest_data_types


####################### Testing the Columns of Dataframe #######################

@pytest.fixture
def datatest_columns() -> List[str]:
    """
    Get the columns of the datatest
    """

    datatest = get_datatest()
    data_columns = datatest.columns.tolist()

    return data_columns


def test_datatest_columns(datatest_columns: Callable) -> NoReturn:
    """
    Test if the columns of the transformed dataframe
    match the columns of the database
    """

    database_schema_columns = ['Product_Code',
                               'Date',
                               'Order_Demand']

    assert database_schema_columns[1] == datatest_columns[1]
