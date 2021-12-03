def bar_poster(chart_df, bar_df, period):  # summary poster
    # Make Subplots
    fig = make_subplots(
        rows=4, cols=1,
        # column_widths=[1],
        specs=[[{'type': 'bar'}],
               [{'type': 'bar'}],
               [{'type': 'bar'}],
               [{'type': 'bar'}]],
        subplot_titles=('X Class Demand by Month',
                        'Y Class Demand by Month',
                        'Z Class Demand by Month',
                        'Top 15 X Class Product'),
        vertical_spacing=0.15, horizontal_spacing=0.3)

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

    # Bar chart 1
    # X class data
    X_data = data_unstacked[data_unstacked['xyz_class'] == 'X']
    X_x = X_data.month
    X_y = X_data.demand

    fig.add_trace(go.Bar(x=X_x, y=X_y,
                         legendgroup='grp1',
                         showlegend=False),
                  row=1, col=1)

    fig.update_traces(hoverinfo='x+y',
                      marker=dict(color=color_map,
                                  line=dict(color='white', width=1)),
                      row=1, col=1)

    # Bar chart 2
    # Y class data
    Y_data = data_unstacked[data_unstacked['xyz_class'] == 'Y']
    Y_x = Y_data.month
    Y_y = Y_data.demand

    fig.add_trace(go.Bar(x=Y_x, y=Y_y,
                         legendgroup='grp1',
                         showlegend=False),
                  row=2, col=1)

    fig.update_traces(hoverinfo='x+y',
                      marker=dict(color=color_map,
                                  line=dict(color='white', width=1)),
                      row=2, col=1)

    fig.update_yaxes(title_text="Product Demand Volume", row=2, col=1)

    # Bar chart 3
    # Z class data
    Z_data = data_unstacked[data_unstacked['xyz_class'] == 'Z']
    Z_x = Z_data.month
    Z_y = Z_data.demand

    fig.add_trace(go.Bar(x=Z_x, y=Z_y,
                         legendgroup='grp1',
                         showlegend=False),
                  row=3, col=1)

    fig.update_traces(hoverinfo='x+y',
                      marker=dict(color=color_map,
                                  line=dict(color='white', width=1)),
                      row=3, col=1)

    # fig.update_yaxes(title_text="Product Demand Volume", row=3, col=1)

    # Bar chart 4
    top_X = bar_df[bar_df['class'] == 'X']
    top_X = top_X.groupby('Product_Code')['Order_Demand'].sum(
    ).sort_values(ascending=False).reset_index()
    top_15 = top_X[:15]
    x_top = top_15.Product_Code
    y_top = top_15.Order_Demand

    fig.add_trace(go.Bar(x=x_top, y=y_top,
                         legendgroup='grp1',
                         showlegend=False),
                  row=4, col=1)

    fig.update_traces(hoverinfo='x+y',
                      marker=dict(color='lightsteelblue',
                                  line=dict(color='white', width=1)),
                      row=4, col=1)

    # fig.update_yaxes(title_text="Product Demand Volume", row=4, col=1)

    fig.update_layout(width=480, height=600)

    return fig
