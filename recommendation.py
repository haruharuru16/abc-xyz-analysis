import pandas as pd
import numpy as np
from helper import abc_classification


def get_recommendation(data, a_input, b_input):
    # Change Date into datetime format
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

    # filtering the data based months
    month_1 = data.loc[(data['Date'].dt.month == 10)]
    month_2 = data.loc[(data['Date'].dt.month != 12)]
    month_3 = data

    # getting the abc classification
    # first month
    month_1_data_abc, month_1_abc_class = abc_classification(
        month_1, a_input, b_input)

    # first and second month
    month_2_data_abc, month_2_abc_class = abc_classification(
        month_2, a_input, b_input)

    # 3 months data
    month_3_data_abc, month_3_abc_class = abc_classification(
        month_3, a_input, b_input)

    # filtering features and renaming each month classes column name
    month_1_data_abc = month_1_data_abc.drop(
        columns={'order_frequency', 'rank', 'total', 'rank_cumsum'})
    month_1_data_abc = month_1_data_abc.rename({'class': 'class_1'}, axis=1)

    month_2_data_abc = month_2_data_abc.drop(
        columns={'order_frequency', 'rank', 'total', 'rank_cumsum'})
    month_2_data_abc = month_2_data_abc.rename({'class': 'class_2'}, axis=1)

    month_3_data_abc = month_3_data_abc.drop(
        columns={'order_frequency', 'rank', 'total', 'rank_cumsum'})
    month_3_data_abc = month_3_data_abc.rename({'class': 'class_3'}, axis=1)

    # Merging Data
    month_3_data_abc = month_3_data_abc.merge(
        month_2_data_abc, how='left', on='Product_Code').fillna('No class')
    month_3_data_abc = month_3_data_abc.merge(
        month_1_data_abc, how='left', on='Product_Code').fillna('No class')

    columns = ['Product_Code', 'class_1', 'class_2', 'class_3']

    month_merge = month_3_data_abc[columns]

    # defining conditions and choices
    conditions = [(month_merge['class_3'] == month_merge['class_2']) & (month_merge['class_3'] == month_merge['class_1']),
                  (((month_merge['class_3'] == month_merge['class_2']) & (month_merge['class_3'] != month_merge['class_1'])) | (month_merge['class_1'] == month_merge['class_2']) & (month_merge['class_1'] != month_merge['class_3']) & (month_merge['class_1'] != 'No class'))]

    choices = ['3 months', '2 months']

    # making the recommendation
    month_merge['Reevaluate period'] = np.select(
        conditions, choices, default='Every month')

    month_merge = month_merge.loc[:, [
        'Product_Code', 'Reevaluate period']]

    return month_merge
