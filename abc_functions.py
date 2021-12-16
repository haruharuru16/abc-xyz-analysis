from threading import Condition
from numpy.core.fromnumeric import product
from numpy.lib.arraypad import pad
from numpy.lib.shape_base import column_stack
from pandas.core.reshape.pivot import pivot
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from streamlit.util import index_


def load_css(filename):
    with open(filename) as f:
        st.markdown('<style>{}</style>'.format(f.read()),
                    unsafe_allow_html=True)


def load_data(filename):  # load data function
    df = pd.read_csv(filename)
    return df


def filter_dataset(data, period):  # filter dataset for XYZ Classification Function
    # copy the data first then change Date into datetime format
    data_copy = data.copy()
    data_copy['Date'] = pd.to_datetime(data_copy['Date'], dayfirst=True)
    data_copy['month'] = data_copy['Date'].dt.month

    # get the last 3 months data
    months = data_copy.month.unique()
    last_3_months = months[-3:]

    # filtering the data based on period choice
    if period == 'Last 3 Months':
        data_period = data_copy.loc[data_copy['month'].apply(
            lambda x: x in last_3_months)]
    else:
        data_period = data_copy

    data_period.drop(columns='month', inplace=True)

    return data_period


def abc_classification(data, a_input, b_input):  # ABC Classification Function
    sum_perc = a_input + b_input

    # Defining ABC Class
    def abc(percentage):
        if percentage >= 0 and percentage <= a_input:
            return 'A'
        elif percentage > a_input and percentage <= sum_perc:
            return 'B'
        else:
            return 'C'

    # Calculate reorder data
    reorder_data = data['Product_Code'].value_counts().reset_index()
    reorder_data = reorder_data.rename(
        {'index': 'Product_Code', 'Product_Code': 'order_frequency'}, axis=1)

    # Calculating product rank based on product reorder freq
    product_calc = reorder_data.copy()
    product_calc['rank'] = product_calc.index + 1

    # Calculating Total Product
    product_calc['total'] = len(product_calc)

    # Calculating Rank Cumulative Sum
    product_calc['rank_cumsum'] = (
        product_calc['rank'] / product_calc['total']) * 100

    # Applying ABC Classification
    product_calc['class'] = product_calc['rank_cumsum'].apply(abc)

    data_abc = product_calc.copy()
    abc_merge = merge_data(product_calc, data)

    return data_abc, abc_merge


def merge_data(data1, data2):  # merge data function
    classified = data1.loc[:, ['Product_Code', 'class']]
    merged = data2.merge(classified, how='left', on='Product_Code')

    return merged


def xyz(cov):  # defining xyz classes
    if cov <= 0.5:
        return 'X'
    elif cov > 0.5 and cov <= 1.0:
        return 'Y'
    else:
        return 'Z'


def xyz_classification(data_period):  # XYZ Classification Function
    # add month column
    data_period['month'] = data_period.Date.apply(lambda x: x.strftime('%B'))

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
    data_calc.sort_values(by='cov', ascending=True, inplace=True)

    # defining xyz classes
    data_xyz = data_calc.copy()
    data_xyz['xyz_class'] = data_xyz['cov'].apply(xyz)

    xyz_monthly = data_xyz.drop(
        columns={'std', 'avg', 'cov'}).groupby('xyz_class').agg('sum')

    xyz_classified = data_xyz.copy().reset_index().rename(
        columns={'xyz_class': 'class'})
    # xyz_classified = xyz_classified.loc[:, ['Product_Code', 'xyz_class']
    xyz_merge = merge_data(xyz_classified, data_period)

    return data_xyz, xyz_monthly, xyz_merge


def abc_xyz_class(data1, data2):
    # process data 1
    data1 = data1.rename({'class': 'abc_class'}, axis=1)
    data1 = data1.groupby(['Product_Code', 'abc_class'])[
        'Order_Demand'].sum().reset_index()

    # process data 2
    data2 = data2.rename({'class': 'xyz_class'}, axis=1)
    data2 = data2.groupby(['Product_Code', 'xyz_class'])[
        'Order_Demand'].sum().reset_index()
    data2 = data2.drop(columns={'Order_Demand'})

    # merge data
    merged = data1.merge(data2, how='left', on='Product_Code')
    merged = merged[['Product_Code', 'Order_Demand', 'abc_class', 'xyz_class']]

    merged = merged.reset_index()
    merged['No'] = merged.index + 1
    merged.drop(columns='index', inplace=True)
    merged.set_index('No', inplace=True)

    return merged


def summary_poster(chart_df):  # summary poster
    # Make Subplots
    fig = make_subplots(
        rows=2, cols=3,
        column_widths=[3, 2, 2],
        specs=[[{'type': 'bar'}, {'type': 'pie'}, {'type': 'pie'}],
               [{'type': 'bar'}, {'type': 'pie'}, {'type': 'table'}]],
        subplot_titles=('ABC Class by Volume Bar Chart',
                        'ABC Class by Percentage',
                        'ABC Class by Volume Percentage',
                        'Top 15 A Class Products Bar Chart',
                        'Top 15 A Class Products',
                        'Top 15 A Class Products Volume by Percentage'),
        vertical_spacing=0.1, horizontal_spacing=0.05)

    color_map = ['darkblue', 'royalblue', 'cyan']

    # PIE
    # data for pie ABC Class by Percentage
    pie_data = chart_df.groupby(
        'class')['Product_Code'].nunique().reset_index()
    pie_data = pie_data.rename(
        columns={'class': 'label', 'Product_Code': 'class_counts'})

    fig.add_trace(go.Pie(labels=pie_data.label,
                         values=pie_data.class_counts,
                         hole=0.4,
                         legendgroup='grp1',
                         showlegend=False),
                  row=1, col=2)

    fig.update_traces(hoverinfo='label+percent',
                      marker=dict(colors=color_map,
                                  line=dict(color='white', width=1)),
                      row=1, col=2)

    # data for pie ABC Class by Volume Percentage
    pie_demand = chart_df.groupby('class')['Order_Demand'].sum().reset_index()
    pie_demand = pie_demand.rename(
        columns={'class': 'label', 'Order_Demand': 'order_volume'})
    # pie 2
    fig.add_trace(go.Pie(labels=pie_demand.label,
                         values=pie_demand.order_volume,
                         hole=0.4,
                         legendgroup='grp1',
                         showlegend=False),
                  row=1, col=3)

    fig.update_traces(hoverinfo='label+percent+value',
                      marker=dict(colors=color_map,
                                  line=dict(color='white', width=1)),
                      row=1, col=3)

    # Bar chart for ABC Classes by Volume
    bar_plot = pie_demand.copy()
    x = bar_plot.label
    y = bar_plot.order_volume
    # bar chart 1
    fig.add_trace(go.Bar(x=x, y=y,
                         legendgroup='grp2',
                         showlegend=False),
                  row=1, col=1)

    fig.update_traces(hoverinfo='x+y',
                      text=y,
                      texttemplate='%{text:.2s}',
                      textposition='inside',
                      marker=dict(color=color_map,
                                  line=dict(color='white', width=1)),
                      row=1, col=1)

    fig.update_yaxes(title_text="Product Demand Volume", row=1, col=1)

    # Bar chart 4 Top 15 A Class Products
    top_A = chart_df[chart_df['class'] == 'A']
    top_A = top_A.groupby('Product_Code')['Order_Demand'].sum(
    ).sort_values(ascending=False).reset_index()
    top_15 = top_A[:15]
    x_top = top_15.Product_Code
    y_top = top_15.Order_Demand
    # bar chart 2
    fig.add_trace(go.Bar(x=x_top, y=y_top,
                         legendgroup='grp2',
                         showlegend=False),
                  row=2, col=1)

    fig.update_traces(hoverinfo='x+y',
                      marker=dict(color="mediumslateblue",
                                  line=dict(color='white', width=1)),
                      row=2, col=1)

    fig.update_yaxes(title_text="Product Demand Volume", row=2, col=1)

    fig.update_xaxes(tickangle=-45, row=2, col=1)

    # Pie Chart Top 15 A Class Products
    aggregate = chart_df.groupby(['Product_Code', 'class'])[
        'Order_Demand'].sum().reset_index()
    # plus_demand = chart_df.merge(aggregate, how='left', on='Product_Code')
    # plus_demand = plus_demand.loc[:, ['Product_Code', 'Order_Demand', 'class']]
    data_a = aggregate.loc[aggregate['class'].apply(
        lambda x: x == 'A')].sort_values('Order_Demand', ascending=False)
    data_a_15 = data_a[:15]
    list_15 = data_a_15['Product_Code'].unique()
    data_others = aggregate.loc[~(
        aggregate['Product_Code'].apply(lambda x: x in list_15))]
    total_demand = sum(data_others['Order_Demand'])
    others = {'Product_Code': 'Others',
              'Order_Demand': total_demand, 'class': 'Others'}
    data_a_15 = data_a_15.append(others, ignore_index=True)
    data_a_15 = data_a_15.rename(
        columns={'class': 'label', 'Order_Demand': 'order_volume'})

    # pie 3
    fig.add_trace(go.Pie(labels=data_a_15.Product_Code,
                         values=data_a_15.order_volume,
                         hole=0.4,
                         legendgroup='grp1',
                         showlegend=False),
                  row=2, col=3)

    fig.update_traces(hoverinfo='label+percent+value',
                      marker=dict(colors=px.colors.sequential.Inferno,
                                  line=dict(color='white', width=1)),
                      row=2, col=3)

    # ABC Classification Table
    fig.add_trace(go.Table(header=dict(values=['Product Code', 'Order Demand', 'Class'],
                                       align='left'),
                           cells=dict(values=[data_a_15.Product_Code, data_a_15.order_volume, data_a_15.label],
                                      align='left')),
                  row=2, col=2)

    fig.update_layout(width=1200, height=600,
                      margin=dict(l=0, r=10, t=20, b=0))

    return fig


def xyz_summary(chart_df, bar_df, period):
    # Make Subplots
    fig = make_subplots(
        rows=2, cols=1,
        # column_widths=[1],
        specs=[[{'type': 'bar'}],
               [{'type': 'bar'}]],
        subplot_titles=('XYZ Class Demand by Month',
                        'Top 15 X Class Product'),
        vertical_spacing=0.1, horizontal_spacing=0)

    # filtering the data based on period choice
    if period == 'First 2 Months':
        data = chart_df.loc[:, ['October', 'November']]
    elif period == 'Last 2 Months':
        data = chart_df.loc[:, ['November', 'December']]
    else:
        data = chart_df.loc[:, ['October', 'November', 'December']]

    # data
    # data = chart_df.drop(columns={'total'}, axis=1)
    data_unstacked = data.unstack().reset_index().rename(columns={0: 'demand'})

    color_map = ['#9D5C0D', '#E5890A', '#F7D08A']

    # Grouped bar charts
    X_data = data_unstacked[data_unstacked['xyz_class'] == 'X']
    Y_data = data_unstacked[data_unstacked['xyz_class'] == 'Y']
    Z_data = data_unstacked[data_unstacked['xyz_class'] == 'Z']

    fig.add_trace(go.Bar(x=X_data.month,
                         y=X_data.demand,
                         text=X_data.demand,
                         texttemplate='%{text:.2s}',
                         textposition='inside',
                         name='X',
                         marker_color='crimson'))
    fig.add_trace(go.Bar(x=Y_data.month,
                         y=Y_data.demand,
                         text=Y_data.demand,
                         texttemplate='%{text:.2s}',
                         textposition='outside',
                         name='Y',
                         marker_color='indianred'))
    fig.add_trace(go.Bar(x=Z_data.month,
                         y=Z_data.demand,
                         text=Z_data.demand,
                         texttemplate='%{text:.2s}',
                         textposition='outside',
                         name='Z',
                         marker_color='salmon'))

    fig.update_layout(barmode='group')

    # Bar chart 2 (top 15 X Class)
    top_X = bar_df[bar_df['class'] == 'X']
    top_X = top_X.groupby('Product_Code')['Order_Demand'].sum(
    ).sort_values(ascending=False).reset_index()
    top_15 = top_X[:15]
    x_top = top_15.Product_Code
    y_top = top_15.Order_Demand

    fig.add_trace(go.Bar(x=x_top, y=y_top,
                         legendgroup='grp1',
                         showlegend=False),
                  row=2, col=1)

    fig.update_traces(hoverinfo='x+y',
                      marker=dict(color='lightsalmon',
                                  line=dict(color='white', width=1)),
                      row=2, col=1)

    fig.update_xaxes(tickangle=-45, row=2, col=1)

    # fig.update_yaxes(title_text="Product Demand Volume", row=4, col=1)

    fig.update_layout(width=480, height=600,
                      margin=dict(l=10, t=20, b=0))

    return fig
