import base64
import io

import dash
import numpy as np
from dash import dcc, html
from dash.dash_table import DataTable
from dash.dependencies import Input, Output
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from EDA import *

# Assume df_train is your DataFrame
# Example DataFrame creation:
df_train = pd.read_csv('train.csv')
# Create the box plots
# Create actual vs predicted price graph for each model
def create_model_graph(model, name):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    fig = go.Figure()

    # Add actual vs predicted points for training set
    fig.add_trace(go.Scatter(
        x=y_train,
        y=y_pred_train,
        mode='markers',
        name=f'{name} (Training)',
        marker=dict(color='blue'),
        text=f'Training RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}',
    ))

    # Add actual vs predicted points for testing set
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred_test,
        mode='markers',
        name=f'{name} (Testing)',
        marker=dict(color='red'),
        text=f'Testing RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}',
    ))

    fig.add_trace(go.Scatter(
        x=[min(y_train), max(y_train)],
        y=[min(y_train), max(y_train)],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Ideal Line'
    ))

    fig.update_layout(
        title=f'Actual vs Predicted Prices - {name}',
        xaxis_title='Actual Prices',
        yaxis_title='Predicted Prices',
        showlegend=True,
        legend=dict(
            x=0.1,
            y=0.9,
            traceorder='normal'
        )
    )

    return fig

def create_box_plots():
    fig = go.Figure()

    # Box plot for OverallQual vs SalePrice
    fig.add_trace(go.Box(
        y=df_train['SalePrice'],
        x=df_train['OverallQual'],
        name='OverallQual vs SalePrice'
    ))

    # Box plot for OverallCond vs SalePrice
    fig.add_trace(go.Box(
        y=df_train['SalePrice'],
        x=df_train['OverallCond'],
        name='OverallCond vs SalePrice'
    ))

    fig.update_layout(
        title='Relationship with categorical features',
        xaxis=dict(
            title='Categorical Features',
            showgrid=False
        ),
        yaxis=dict(
            title='SalePrice',
            showgrid=False
        ),
        boxmode='group'
    )

    return fig


def create_corr_heatmap():
    # Exclude non-numeric columns
    numeric_columns = df_train.select_dtypes(include=['number']).columns
    corrmat = df_train[numeric_columns].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corrmat.values,
        x=corrmat.columns,
        y=corrmat.columns,
        colorscale='Viridis',
        zmin=-1,
        zmax=1
    ))

    fig.update_layout(
        title='Correlation Matrix',
        xaxis=dict(
            title='Features'
        ),
        yaxis=dict(
            title='Features'
        )
    )

    return fig


# Create the heatmap of the correlation matrix for top variables correlated with 'SalePrice'
def create_saleprice_corr_heatmap():
    # Filter out non-numeric columns
    numeric_columns = df_train.select_dtypes(include=['number']).columns
    df_numeric = df_train[numeric_columns]

    # Calculate correlation matrix
    k = 10  # Number of variables for heatmap
    corrmat = df_numeric.corr()
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df_numeric[cols].values.T)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=cols,
        y=cols,
        colorscale='Viridis',
        zmin=-1,
        zmax=1
    ))

    fig.update_layout(
        title='Top Variables Correlated with SalePrice',
        xaxis=dict(
            title='Features'
        ),
        yaxis=dict(
            title='Features'
        )
    )

    return fig
# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='DataFrame Description'),

    dbc.Row([
        dbc.Col(
            DataTable(
                id='description-table',
                columns=[{'name': col, 'id': col} for col in df_train.describe().columns],
                data=df_train.describe().to_dict('records'),
             style_table = {'width': '100px'}  # Set the width to 100%
            ),
            width=3
        ),
        dbc.Col(
            html.Div(
                id='columns-text',
                children=[html.Span(col, style={'margin-right': '5px', 'font-family': 'Arial'}) for col in df_train.columns],
                style={'display': 'flex', 'flex-wrap': 'wrap', 'align-items': 'center','width': '500px', 'height': '500px',}
            ),
            width=9
        )
    ]),
    dbc.Row([
                html.H1(children='SalePrice Distribution Plot'),

                # Display the distribution plot
                dcc.Graph(
                    id='saleprice-distplot',
                    figure=px.histogram(df_train, x='SalePrice', nbins=20, title='SalePrice Distribution')
                ),
                html.Div(id='distribution-characteristics'),
                html.H1(children='Relationship between Numerical Variables'),

                    # Display the scatter plots
                dcc.Graph(
                    id='scatter-plots',
                ),
        html.H1(children='Box Plots'),

        dcc.Graph(
            id='box-plots',
            figure=create_box_plots()
        ),
        dcc.Graph(
            id='corr-heatmap',
            figure=create_corr_heatmap()
        ),
        dcc.Graph(
            id='saleprice-corr-heatmap',
            figure=create_saleprice_corr_heatmap()
        ),
        dcc.Graph(id='scatter-plot'),
        html.H1(children='Actual vs Predicted Prices for Different Models'),

        # Display the actual vs predicted price graph for each model
        html.Div(id='model-graphs'),



        # Display model scores
        html.Div(id='model-scores')

    ]),
])

# Calculate distribution characteristics
def calculate_characteristics():
    skewness = df_train['SalePrice'].skew()
    kurtosis = df_train['SalePrice'].kurtosis()
    characteristics_text = f"Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}"
    return characteristics_text

# Callback to update the distribution characteristics text
@app.callback(
    Output('distribution-characteristics', 'children'),
    Input('saleprice-distplot', 'figure')
)
def update_characteristics(figure):
    characteristics_text = calculate_characteristics()
    return html.Div(characteristics_text)


def create_scatter_plots():
    fig = make_subplots(rows=1, cols=2, subplot_titles=('GrLivArea vs SalePrice', 'TotalBsmtSF vs SalePrice'))

    # Scatter plot for GrLivArea vs SalePrice
    trace1 = go.Scatter(
        x=df_train['GrLivArea'],
        y=df_train['SalePrice'],
        mode='markers',
        marker=dict(color='blue'),
        name='GrLivArea vs SalePrice'
    )
    fig.add_trace(trace1, row=1, col=1)

    # Scatter plot for TotalBsmtSF vs SalePrice
    trace2 = go.Scatter(
        x=df_train['TotalBsmtSF'],
        y=df_train['SalePrice'],
        mode='markers',
        marker=dict(color='red'),
        name='TotalBsmtSF vs SalePrice'
    )
    fig.add_trace(trace2, row=1, col=2)

    fig.update_xaxes(title_text='GrLivArea', row=1, col=1)
    fig.update_yaxes(title_text='SalePrice', row=1, col=1)
    fig.update_xaxes(title_text='TotalBsmtSF', row=1, col=2)
    fig.update_yaxes(title_text='SalePrice', row=1, col=2)

    fig.update_layout(showlegend=False)

    return fig


# Callback to update the scatter plots
@app.callback(
    Output('scatter-plots', 'figure'),
    Input('scatter-plots', 'id')
)
def update_plots(value):
    fig = create_scatter_plots()
    return fig
# Create the box plots
def create_box_plots():
    fig = go.Figure()

    # Box plot for OverallQual vs SalePrice
    fig.add_trace(go.Box(
        y=df_train['SalePrice'],
        x=df_train['OverallQual'],
        name='OverallQual vs SalePrice'
    ))

    # Box plot for OverallCond vs SalePrice
    fig.add_trace(go.Box(
        y=df_train['SalePrice'],
        x=df_train['OverallCond'],
        name='OverallCond vs SalePrice'
    ))

    fig.update_layout(
        title='Relationship with categorical features',
        xaxis=dict(
            title='Categorical Features',
            showgrid=False
        ),
        yaxis=dict(
            title='SalePrice',
            showgrid=False
        ),
        boxmode='group'
    )

    return fig


# Callback to update model graphs
@app.callback(
    Output('model-graphs', 'children'),
    Input('actual-predicted-graph', 'figure')  # Assuming you have an actual-predicted-graph for another purpose
)
def update_model_graphs(figure):
    model_graphs = []

    for model, name in zip([rf_model, svr_model, gb_model, lr_model], ['Random Forest', 'Support Vector', 'Gradient Boosting', 'Linear Regression']):
        model_fig = create_model_graph(model, name)
        model_graphs.append(dcc.Graph(figure=model_fig))

    return model_graphs
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
