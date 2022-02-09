from calendar import month
import pandas as pd
import numpy as np


def get_rec(data_period):

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
