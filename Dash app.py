import subprocess
import sys
import pandas as pd
import umap
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import numpy as np

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = ["umap-learn", "pandas", "plotly", "scikit-learn", "dash"]

# Attempt to import required packages, install if they're missing
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# Function to create UMAP plot and return figure along with scaled data
def create_umap_plot(data_scaled, n_neighbors, min_dist, spread, metric):
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, spread=spread,
                           n_components=3, metric=metric, random_state=42)
    umap_results = umap_model.fit_transform(data_scaled)

    fig = go.Figure(data=[go.Scatter3d(
        x=umap_results[:, 0], y=umap_results[:, 1], z=umap_results[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue'),
        selected={'marker': {'color': 'red'}},
        unselected={'marker': {'opacity': 0.3}},
        )])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), dragmode='lasso')
    return fig, umap_results

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='umap-plot'),
    dash_table.DataTable(id='selected-data', columns=[]),
])

@app.callback(
    [Output('umap-plot', 'figure'),
     Output('selected-data', 'columns'),
     Output('selected-data', 'data')],
    [Input('umap-plot', 'selectedData')]
)
def update_output(selectedData):
    # Load and prepare the dataset
    data = pd.read_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/test/AD.csv')
    data_for_umap = data.iloc[:, 5:]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_umap)

    # Create UMAP plot with the specified parameters
    fig, umap_results = create_umap_plot(data_scaled, 15, 0.1, 1.0, 'euclidean')

    # Prepare data for the data table based on selected points
    selected_indices = [point['pointIndex'] for point in selectedData['points']] if selectedData else []
    selected_data = data.iloc[selected_indices] if selected_indices else pd.DataFrame(columns=data.columns)
    columns = [{"name": i, "id": i} for i in selected_data.columns]

    return fig, columns, selected_data.to_dict('records')

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
