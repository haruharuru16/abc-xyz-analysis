from threading import Condition
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from matplotlib import rcParams

st.title('ABC-XYZ Classification Dashboard')

# load dataset
dataset = pd.read_csv('Product Demand 3 Months.csv')
dataset = pd.DataFrame(dataset)
dataset = dataset.drop(columns='Unnamed: 0')
# st.dataframe(dataset)
# st.write(dataset.columns)

# Calculating Product Reorder Frequency
reorder_frequency = pd.DataFrame(
    dataset['Product_Code'].value_counts().reset_index())
reorder_frequency = reorder_frequency.rename(
    {'index': 'Product_Code', 'Product_Code': 'order_frequency'}, axis=1)

# Calculating Product Rank Based on Product Reorder Frequency
product_calc = reorder_frequency.copy()
product_calc['rank'] = product_calc.index + 1

# Calculating Total Product
product_calc['total'] = len(product_calc)

# Calculating Rank Cumulative Sum
product_calc['rank_cumsum'] = round(
    ((product_calc['rank'] / product_calc['total']) * 100), 2)

# Input A and B Class Percentage
a_percentage = st.sidebar.number_input('Input A Class Products Percentage')
b_percentage = st.sidebar.number_input('Input B Class Products Percentage')
sum_b_percentage = a_percentage + b_percentage

# Defining ABC Classification


def abc(percentage):
    if percentage >= 0 and percentage < a_percentage:
        return 'A'
    elif percentage > a_percentage and percentage <= sum_b_percentage:
        return 'B'
    else:
        return 'C'


# Applying ABC Classification
product_calc['class'] = product_calc['rank_cumsum'].apply(abc)
# st.dataframe(product_calc)

# Getting The Classification
class_counts = pd.DataFrame(
    product_calc['class'].value_counts().sort_values().reset_index())
class_counts = class_counts.rename(
    columns={'index': 'class', 'class': 'class_counts'})
# st.dataframe(class_counts)

# Getting Order Frequency and Class
classified = product_calc.loc[:, ['Product_Code', 'order_frequency', 'class']]
# st.dataframe(classified)

# Getting Order Frequency Total
demand_freq = pd.DataFrame(classified.groupby('class')[
                           'order_frequency'].sum())
demand_freq['class_counts'] = product_calc['class'].value_counts()
demand_freq = demand_freq.reset_index()
demand_freq = demand_freq[['class', 'class_counts', 'order_frequency']]

# Visualizing the Demand Frequency
sns.set_style('whitegrid')
bar, ax = plt.subplots(figsize=(10, 6))
ax = sns.barplot(x=demand_freq['class'], y=demand_freq['order_frequency'],
                 data=demand_freq, ci=None, palette='muted', orient='v')
ax.set_title('ABC Class Order Demand Frequency Total', fontsize=15)
ax.set_xlabel('Class')
ax.set_ylabel('Demand Frequency')
labels, locations = plt.yticks()
plt.yticks(labels, (labels/1).astype(int))

# st.pyplot(bar)
# st.dataframe(demand_freq)

# Merging The Data
merge = dataset.merge(classified, how='left', on='Product_Code')

# Getting the Class Percentage
class_perc = merge.groupby('class')['Product_Code'].nunique()

# Visualizing the class Percentation
pie, ax = plt.subplots(figsize=[7, 7])
labels = class_perc.keys()
colors = sns.color_palette('pastel')[0:5]
plt.pie(x=class_perc, autopct='%.1f%%',
        colors=colors, explode=[0.05]*3,
        labels=labels, pctdistance=0.6)
plt.title('ABC Class by Percentage', fontsize=8)
st.pyplot(pie)

# Calculating Demand Percentage
demand_perc = merge.groupby('class')['Order_Demand'].sum()

# Visualizing the Demand Percentage
pie, ax = plt.subplots(figsize=[10, 6])
labels = demand_perc.keys()
colors = sns.color_palette('pastel')[0:5]
plt.pie(x=demand_perc, autopct='%.1f%%',
        colors=colors, explode=[0.05]*3,
        labels=labels, pctdistance=0.6)
plt.title('ABC Class Demand Volume by Percentage', fontsize=15)
st.pyplot(pie)


# Performing XYZ Classification
data_month = dataset.copy()
data_month['Date'] = pd.to_datetime(data_month['Date'])

# Extracting the month
data_month['month'] = data_month.Date.apply(lambda x: x.strftime('%B'))

# Grouping data by Product Code and Month
data_group = pd.DataFrame(data_month.groupby(
    ['Product_Code', 'month'])['Order_Demand'].sum())
data_group = data_group.reset_index()

# Making Data Pivot
data_pivot = data_group.copy()
data_pivot = data_pivot.pivot(
    index='Product_Code', columns='month', values='Order_Demand')
data_pivot = data_pivot.fillna(0)

# Reformating the data
months = ['October', 'November', 'December']
data_pivot = data_pivot[months]

# Calculating Demand Standard Deviation
data_calc = data_pivot.copy()
data_calc['std'] = round((data_calc[months].std(axis=1)), 2)

# Calculating Total Demand within Period Given
data_calc['total'] = round((data_calc[months].sum(axis=1)), 2)

# Calculating Average Demand within Period Given
data_calc['avg'] = round((data_calc[months].mean(axis=1)), 2)

# Calculating Covariance Demand
data_calc['cov'] = round((data_calc['std'] / data_calc['avg']), 2)

# Sorting the data
data_calc.sort_values(by='cov', ascending=True, inplace=True)

# Visualizing the Coefficient of Demand Variation
f, ax = plt.subplots(figsize=(15, 6))
ax = sns.histplot(data_calc['cov'], kde=True)
plt.title('Coefficient of Demand Variation Distribution', fontsize=20, pad=20)
plt.xlabel('Covariance Demand', fontsize=15)
plt.ylabel('Counts', fontsize=15)
# st.pyplot(f)

# Classifying XYZ classes


def xyz(cov):
    if cov <= 0.5:
        return 'X'
    elif cov > 0.5 and cov <= 1.0:
        return 'Y'
    else:
        return 'Z'


# Applying the xyz classes
data_xyz = data_calc.copy()
data_xyz['xyz_class'] = data_xyz['cov'].apply(xyz)

# Getting statistics
xyz = data_xyz.copy()
xyz.groupby('xyz_class').agg(
    total_demand=('total', 'sum'),
    std_demand=('std', 'mean'),
    avg_demand=('avg', 'mean'),
    avg_cov=('cov', 'mean')
)

# Getting monthly xyz demand
xyz_monthly = data_xyz.groupby('xyz_class').agg(
    oct=('October', 'sum'),
    nov=('November', 'sum'),
    dec=('December', 'sum')
)

# Getting unstacked data
xyz_monthly_unstacked = xyz_monthly.unstack('xyz_class').to_frame()
xyz_monthly_unstacked = xyz_monthly_unstacked.reset_index().rename(
    columns={'level_0': 'month', 0: 'demand'})

# Visualizing the data
# X data
X, ax = plt.subplots(figsize=(15, 6))
ax = sns.barplot(x='month',
                 y='demand',
                 data=xyz_monthly_unstacked[xyz_monthly_unstacked['xyz_class'] == 'X'],
                 palette='Blues_d')
plt.title('X Class Demand by Month', fontsize=20, pad=20)
plt.xlabel('Month', fontsize=15)
plt.ylabel('Demand (in million)', fontsize=15)
labels, locations = plt.yticks()
plt.yticks(labels, (labels/1000000).astype(int))
st.pyplot(X)

# Y data
Y, ax = plt.subplots(figsize=(15, 6))
ax = sns.barplot(x='month',
                 y='demand',
                 data=xyz_monthly_unstacked[xyz_monthly_unstacked['xyz_class'] == 'Y'],
                 palette='Blues_d')
plt.title('Y Class Demand by Month', fontsize=20, pad=20)
plt.xlabel('Month', fontsize=15)
plt.ylabel('Demand (in million)', fontsize=15)
labels, locations = plt.yticks()
plt.yticks(labels, (labels/1000000).astype(int))
st.pyplot(Y)

# Z data
Z, ax = plt.subplots(figsize=(15, 6))
ax = sns.barplot(x='month',
                 y='demand',
                 data=xyz_monthly_unstacked[xyz_monthly_unstacked['xyz_class'] == 'Z'],
                 palette='Blues_d')
plt.title('Z Class Demand by Month', fontsize=20, pad=20)
plt.xlabel('Month', fontsize=15)
plt.ylabel('Demand (in million)', fontsize=15)
labels, locations = plt.yticks()
plt.yticks(labels, (labels/1000000).astype(int))
st.pyplot(Z)

# XYZ Data
fxyz, ax = plt.subplots(figsize=(15, 6))
ax = sns.barplot(x='month',
                 y='demand',
                 hue='xyz_class',
                 data=xyz_monthly_unstacked,
                 palette='Blues_d')
plt.title('XYZ Class Demand Monthly', fontsize=20, pad=20)
plt.xlabel('Month', fontsize=15)
plt.ylabel('Demand (in million)', fontsize=15)
labels, locations = plt.yticks()
plt.yticks(labels, (labels/1000000).astype(int))
st.pyplot(fxyz)

# st.dataframe(xyz_monthly_unstacked)

# Performing ABCXYZ Classification
# Load data
xyz_classified = data_xyz.copy().reset_index()
xyz_classified = xyz_classified.loc[:, ['Product_Code', 'xyz_class']]

abc_classified = classified.copy()
abc_classified.drop(columns='order_frequency', inplace=True)
abc_classified = abc_classified.rename({'class': 'abc_class'}, axis=1)

# Merging the data
data_abcxyz = abc_classified.merge(
    xyz_classified, on='Product_Code', how='left')
data_abcxyz['abc_xyz_class'] = data_abcxyz['abc_class'].astype(
    str) + data_abcxyz['xyz_class'].astype(str)
merge_abcxyz = dataset.merge(data_abcxyz, on='Product_Code', how='left')

# Getting the data summary
data_summary = merge_abcxyz.groupby('abc_xyz_class').agg(
    total_demand=('Order_Demand', sum),
    avg_demand=('Order_Demand', 'mean')
).reset_index()
# data_summary['avg_demand'] = round((data_summary['avg_demand']), 2)
# st.dataframe(data_summary)

# Performing Class Recommendation Duration
# Load Data
data_copy = dataset.copy()
data_copy['Date'] = pd.to_datetime(data_copy['Date'])
month_1 = data_copy.loc[(data_copy['Date'].dt.month == 10)]
month_2 = data_copy.loc[(data_copy['Date'].dt.month == 10)
                        | (data_copy['Date'].dt.month == 11)]
month_3 = data_copy.loc[(data_copy['Date'].dt.month == 10) |
                        (data_copy['Date'].dt.month == 11) | (data_copy['Date'].dt.month == 12)]

# Calculating Reorder Frequency for Each Month
sum_1 = pd.DataFrame(month_1['Product_Code'].value_counts().reset_index())
sum_1 = sum_1.rename(
    {'index': 'Product_Code', 'Product_Code': 'order_frequency'}, axis=1)
sum_2 = pd.DataFrame(month_2['Product_Code'].value_counts().reset_index())
sum_2 = sum_2.rename(
    {'index': 'Product_Code', 'Product_Code': 'order_frequency'}, axis=1)
sum_3 = pd.DataFrame(month_3['Product_Code'].value_counts().reset_index())
sum_3 = sum_3.rename(
    {'index': 'Product_Code', 'Product_Code': 'order_frequency'}, axis=1)

# Calculating Rank Products For Each Month Cumulation
# Rank
sum_1['rank'] = sum_1.index + 1
sum_2['rank'] = sum_2.index + 1
sum_3['rank'] = sum_3.index + 1
# Total
sum_1['total'] = len(sum_1)
sum_2['total'] = len(sum_2)
sum_3['total'] = len(sum_3)
# Rank Cumsum
sum_1['rank_cumsum'] = (sum_1['rank'] / sum_1['total']) * 100
sum_2['rank_cumsum'] = (sum_2['rank'] / sum_2['total']) * 100
sum_3['rank_cumsum'] = (sum_3['rank'] / sum_3['total']) * 100

# Defining ABC Class for Each Month Cumulation
sum_1['class_1'] = sum_1['rank_cumsum'].apply(abc)
sum_2['class_2'] = sum_2['rank_cumsum'].apply(abc)
sum_3['class_3'] = sum_3['rank_cumsum'].apply(abc)

# Merging Data
df_1 = sum_1.loc[:, ['Product_Code', 'class_1']]
df_2 = sum_2.loc[:, ['Product_Code', 'class_2']]
df_3 = sum_3.loc[:, ['Product_Code', 'class_3']]

month_merge = pd.merge(df_3, df_2, how='left',
                       on='Product_Code').fillna('No class')
month_merge = pd.merge(month_merge, df_1, how='left',
                       on='Product_Code').fillna('No class')
month_merge = month_merge[['Product_Code', 'class_1', 'class_2', 'class_3']]
# st.dataframe(month_merge)

# Giving recommendation
conditions = [(month_merge['class_3'] == month_merge['class_2']) & (month_merge['class_3'] == month_merge['class_1']),
              ((month_merge['class_3'] == month_merge['class_2']) & (month_merge['class_3'] != month_merge['class_1'])) | (
                  (month_merge['class_1'] == month_merge['class_2']) & (month_merge['class_1'] != month_merge['class_3']) & (month_merge['class_1'] != 'No class'))
              ]
choices = ['Evaluate every 3 months', 'Evaluate every 2 months']

month_copy = month_merge.copy()
month_copy['recommendation'] = np.select(
    conditions, choices, default='Evaluate every month')
show_rec = month_copy.loc[:, ['Product_Code', 'recommendation']]
st.dataframe(show_rec)
