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


def get_xyz_recommendation(data_period):

    # add month column
    data_period['month'] = data_period.Date.dt.month

    # groupby month
    data_group = data_period.groupby(['Product_Code', 'month'])[
        'Order_Demand'].sum().reset_index()

    # pivot the data
    pivot_data = data_group.pivot(
        index='Product_Code', columns='month', values='Order_Demand').fillna(0)

    months = pivot_data.columns

    # calculating demand standard deviation, total demand, average demand, and covariance demand
    data_calc = pivot_data.copy()
    data_calc['std'] = data_calc[months].std(axis=1)
    data_calc['total'] = data_calc[months].sum(axis=1)
    data_calc['avg'] = data_calc[months].mean(axis=1)
    data_calc['cov'] = data_calc['std'] / data_calc['avg']
    data_calc.sort_values(by=['cov'], ascending=True, inplace=True)
    data_calc.reset_index(inplace=True)

    # defining the period recommendation
    def xyz_rec(cov):
        if cov >= 0 and cov <= 0.5:
            return 'Every 3 months'
        elif cov > 0.5 and cov <= 1:
            return 'Every 2 months'
        else:
            return 'Every month'

    # calculating the recommendation
    data_calc['Reevaluate_Period'] = data_calc['cov'].apply(xyz_rec)
    data_rec = data_calc.loc[:, ['Product_Code', 'Reevaluate_Period']]
    data_rec.sort_values(by='Product_Code', ascending=True, inplace=True)
    data_rec.set_index('Product_Code', inplace=True)
    data_rec.reset_index(inplace=True)
    data_rec['No'] = data_rec.index + 1
    data_rec.set_index('No', inplace=True)

    return data_calc, data_rec
