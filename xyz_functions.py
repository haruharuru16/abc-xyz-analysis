import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from streamlit.util import index_
from abc_functions import merge_data


def xyz_cov(cov):  # defining xyz classes
    if cov <= 0.5:
        return 'X'
    elif cov > 0.5 and cov <= 1.0:
        return 'Y'
    else:
        return 'Z'


def xyz_classifier(data_period, x_input, y_input):  # XYZ Classification Function
    sum_perc = x_input + y_input

    # defining xyz classes
    def xyz_classes(cov):
        if cov >= 0 and cov <= x_input:
            return 'X'
        elif cov > x_input and cov <= sum_perc:
            return 'Y'
        else:
            return 'Z'

    # add month column
    # data_period['month'] = data_period.Date.apply(lambda x: x.strftime('%B'))
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
    data_calc.sort_values(by='cov', ascending=True, inplace=True)
    data_calc.reset_index(inplace=True)
    data_calc['rank'] = data_calc.index + 1
    data_calc['total_product'] = len(data_calc)
    data_calc['rank_cumsum'] = (
        data_calc['rank'] / data_calc['total_product']) * 100

    # defining xyz classes
    data_xyz = data_calc.copy()
    data_xyz['xyz_class'] = data_xyz['rank_cumsum'].apply(xyz_classes)

    xyz_monthly = data_xyz.drop(
        columns={'std', 'avg', 'cov', 'rank', 'total_product', 'rank_cumsum'}).groupby('xyz_class').agg('sum')

    xyz_classified = data_xyz.copy().rename(columns={'xyz_class': 'class'})
    # xyz_classified = xyz_classified.loc[:, ['Product_Code', 'class']]
    xyz_merge = merge_data(xyz_classified, data_period)

    return data_xyz, xyz_monthly, xyz_merge


def to_month(month):
    if month == 1:
        return 'January'
    elif month == 2:
        return 'February'
    elif month == 3:
        return 'March'
    elif month == 4:
        return 'April'
    elif month == 5:
        return 'May'
    elif month == 6:
        return 'June'
    elif month == 7:
        return 'July'
    elif month == 8:
        return 'August'
    elif month == 9:
        return 'September'
    elif month == 10:
        return 'October'
    elif month == 11:
        return 'November'
    else:
        return 'December'


def xyz_poster(chart_df, bar_df, period):
    # Make Subplots
    fig = make_subplots(
        rows=2, cols=3,
        column_widths=[3, 2, 2],
        specs=[[{'type': 'bar'}, {'type': 'pie'}, {'type': 'pie'}],
               [{'type': 'bar'}, {'type': 'table'}, {'type': 'domain'}]],
        subplot_titles=('XYZ Class Demand by Month',
                        'XYZ Class by Percentage',
                        'XYZ Class by Volume Percentage',
                        'Top 15 X Class Products Bar Chart',
                        'Top 15 X Class Products',
                        'Top 15 X Class Products by Percentage'),
        vertical_spacing=0.1, horizontal_spacing=0.025)

    # data
    data = chart_df.drop(columns={'total'}, axis=1)
    data_unstacked = data.unstack().reset_index().rename(columns={0: 'demand'})
    data_unstacked['month_string'] = data_unstacked['month'].apply(to_month)

    color_map = ['crimson', 'indianred', 'salmon']

    # Grouped bar charts
    X_data = data_unstacked[data_unstacked['xyz_class'] == 'X']
    Y_data = data_unstacked[data_unstacked['xyz_class'] == 'Y']
    Z_data = data_unstacked[data_unstacked['xyz_class'] == 'Z']

    fig.add_trace(go.Bar(x=X_data.month_string,
                         y=X_data.demand,
                         text=X_data.demand,
                         texttemplate='%{text:.2s}',
                         textposition='inside',
                         name='X',
                         marker_color='crimson'), row=1, col=1)
    fig.add_trace(go.Bar(x=Y_data.month_string,
                         y=Y_data.demand,
                         text=Y_data.demand,
                         texttemplate='%{text:.2s}',
                         textposition='inside',
                         name='Y',
                         marker_color='indianred'), row=1, col=1)
    fig.add_trace(go.Bar(x=Z_data.month_string,
                         y=Z_data.demand,
                         text=Z_data.demand,
                         texttemplate='%{text:.2s}',
                         textposition='outside',
                         name='Z',
                         marker_color='salmon'), row=1, col=1)

    fig.update_layout(yaxis=dict(title='Product Demand Volume'),
                      legend=dict(x=0.40, y=1.0,
                                  bgcolor='rgba(255, 255, 255, 0)',
                                  bordercolor='rgba(255, 255, 255, 0)'),
                      barmode='group')

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

    fig.update_yaxes(title_text='Product Demand Volume', row=2, col=1)

    # pie chart XYZ Class by Percentage
    pie_data = bar_df.groupby('class')['Product_Code'].nunique().reset_index()
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

    # pie chart XYZ Class by Volume Percentage
    pie_demand = bar_df.groupby('class')['Order_Demand'].sum().reset_index()
    pie_demand = pie_demand.rename(
        columns={'class': 'label', 'Order_Demand': 'order_volume'})

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

    # pie chart for Top 15 X Class Products
    aggregate = bar_df.groupby(['Product_Code', 'class'])[
        'Order_Demand'].sum().reset_index()
    data_x = aggregate.loc[aggregate['class'].apply(
        lambda x: x == 'X')].sort_values('Order_Demand', ascending=False)
    data_x_15 = data_x[:15]
    list_15 = data_x_15['Product_Code'].unique()
    data_others = aggregate.loc[~(
        aggregate['Product_Code'].apply(lambda x: x in list_15))]
    total_demand = sum(data_others['Order_Demand'])
    others = {'Product_Code': 'Others',
              'Order_Demand': total_demand, 'class': 'Others'}
    data_x_15 = data_x_15.append(others, ignore_index=True)
    data_x_15 = data_x_15.rename(
        columns={'class': 'label', 'Order_Demand': 'order_volume'})

    fig.add_trace(go.Pie(labels=data_x_15.Product_Code,
                         values=data_x_15.order_volume,
                         hole=0.4,
                         rotation=-45,
                         legendgroup='grp1',
                         showlegend=False),
                  row=2, col=3)
    fig.update_traces(hoverinfo='label+percent+value',
                      marker=dict(colors=px.colors.sequential.Sunsetdark,
                                  line=dict(color='white', width=1)),
                                row=2, col=3)

    # XYZ Classification Table
    fig.add_trace(go.Table(header=dict(values=['Product Code', 'Order Demand', 'Class'],
                                       align='left'),
                           cells=dict(values=[data_x_15.Product_Code, data_x_15.order_volume, data_x_15.label],
                                      align='left')),
                  row=2, col=2)

    # fig.update_yaxes(title_text="Product Demand Volume", row=4, col=1)

    fig.update_layout(width=1200, height=600,
                      margin=dict(l=0, r=10, t=20, b=0))

    return fig
