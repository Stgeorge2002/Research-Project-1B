import subprocess
import sys
import pandas as pd
import numpy as np
import umap.umap_ as umap
import plotly.express as px
from sklearn.preprocessing import StandardScaler

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = ["umap-learn", "pandas", "plotly", "scikit-learn"]

# Attempt to import required packages, install if they're missing
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# Define the datasets to be loaded
datasets = ['ADB.csv']
dataset_names = ['ADB']

# Define the metrics and parameters to iterate over
metrics = ["chebyshev"]
n_neighbors_options = [15]
min_dist_options = [0.]
spread_options = [1]
target_weight_options = [0]  # Corrected to iterate over this list

# Define a color palette
color_palette = px.colors.qualitative.Plotly

for dataset, dataset_name in zip(datasets, dataset_names):
    # Load the dataset
    data = pd.read_csv(f'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/{dataset}')

    # Replace NaN values with zero (assuming the 7th column onwards are features)
    data_for_umap = data.iloc[:, 7:].fillna(0)

    # Scaling the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_umap)

    # Convert ID to categorical numeric labels for UMAP
    ID_labels = data['ID'].astype('category').cat.codes

    for metric in metrics:
        for n_neighbors in n_neighbors_options:
            for min_dist in min_dist_options:
                for spread in spread_options:
                    for target_weight in target_weight_options:  # Iterating over target_weight
                        # Initialize UMAP for 2D with supervised dimension reduction
                        umap_model = umap.UMAP(n_neighbors=n_neighbors,
                                               min_dist=min_dist,
                                               spread=spread,
                                               n_components=2,  # Set to 2 for 2D projection
                                               metric=metric,
                                               n_jobs=1,  # Adjusted to 1 due to random_state override warning
                                               random_state=42,
                                               target_weight=target_weight)  # Now using a single float value

                        # Fit and transform the data using ID as label
                        umap_results = umap_model.fit_transform(data_scaled, y=ID_labels)

                        # Create a new DataFrame for UMAP results
                        umap_df = pd.DataFrame(umap_results, columns=['UMAP-1', 'UMAP-2'])
                        umap_df['ID'] = data['ID'].astype(str)
                        umap_df['Identifier'] = data['Identifier']
                        # Calculate mean coordinates for each ID if necessary (depends on use case)
                        # mean_coords = umap_df.groupby('ID')[['UMAP-1', 'UMAP-2']].mean().reset_index()
                        # Create a color mapping for each unique ID directly from umap_df
                        unique_IDs = umap_df['ID'].unique()
                        color_map = {ID: color_palette[i % len(color_palette)] for i, ID in enumerate(unique_IDs)}
                        # Plot using Plotly for 2D
                        fig = px.scatter(umap_df, x='UMAP-1', y='UMAP-2', color='ID', hover_data=['Identifier'], color_discrete_map=color_map)
                        fig.update_layout(title=f'{dataset_name} - 2D UMAP Projection with {metric}, n_neighbors={n_neighbors}, min_dist={min_dist}, spread={spread}, target_weight={target_weight}, Supervised by ID')

                        # Save Plotly fig as an HTML file for each dataset
                        fig.write_html(f'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/OUTPUT2D/2d_umap_{dataset_name}_{metric}_n_neighbors_{n_neighbors}_min_dist_{min_dist}_spread_{spread}_target_weight_{target_weight}_supervised.html')
