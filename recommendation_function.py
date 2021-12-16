import pandas as pd
import numpy as np
from abc_functions import abc_classification


def get_recommendation(data, a_input, b_input):
    # Change Date into datetime format
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

    # filtering the data based months
    data['month'] = data['Date'].dt.month

    # get the last 3 months data
    months = data.month.unique()
    last_3_months = months[-3:]

    # filtering data
    month_3 = data.loc[data['month'].apply(lambda x: x in last_3_months)]
    month_6 = data

    # getting the abc classification
    # first month
    month_3_data_abc, month_3_abc_class = abc_classification(
        month_3, a_input, b_input)

    # first and second month
    month_6_data_abc, month_6_abc_class = abc_classification(
        month_6, a_input, b_input)

    # filtering features and renaming each month classes column name
    month_3_data_abc = month_3_data_abc.loc[:, ['Product_Code', 'class']]
    month_3_data_abc = month_3_data_abc.rename({'class': 'class_1'}, axis=1)

    month_6_data_abc = month_6_data_abc.loc[:, ['Product_Code', 'class']]
    month_6_data_abc = month_6_data_abc.rename({'class': 'class_2'}, axis=1)

    # # Merging Data
    month_6_data_abc = month_6_data_abc.merge(
        month_3_data_abc, how='left', on='Product_Code').fillna('No class')

    month_merge = month_6_data_abc.loc[:, [
        'Product_Code', 'class_1', 'class_2']]

    # # defining conditions and choices
    conditions = [(month_merge['class_2'] == month_merge['class_1']) & (
        (month_merge['class_1'] != 'No class') | (month_merge['class_2'] != 'No class'))]

    choices = ['6 months']

    # making the recommendation
    month_merge['Reevaluate period'] = np.select(
        conditions, choices, default='Every 3 months')

    month_merge = month_merge.loc[:, [
        'Product_Code', 'Reevaluate period']]

    month_merge['No'] = month_merge.index + 1
    month_merge.set_index('No', inplace=True)

    return month_merge
