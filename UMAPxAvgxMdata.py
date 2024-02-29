import subprocess
import sys
import pandas as pd
import umap.umap_ as umap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = ["umap-learn", "pandas", "plotly", "scikit-learn", "matplotlib"]

# Attempt to import required packages, install if they're missing
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# Define the datasets to be loaded
datasets = ['morphological_data.csv', 'intensity_measurements_data.csv', 'spatial_texture_data.csv', 'ser_analysis_data.csv']
dataset_names = ['Morphological', 'Intensity Measurements', 'Spatial Texture', 'SER Analysis']

# Define the metrics and parameters to iterate over
metrics = ["jaccard"]
n_neighbors_options = [100, 15, 2]
min_dist_options = [0.0]
spread_options = [1]

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

    for metric in metrics:
        for n_neighbors in n_neighbors_options:
            for min_dist in min_dist_options:
                for spread in spread_options:
                    # Initialize UMAP
                    umap_model = umap.UMAP(n_neighbors=n_neighbors,
                                           min_dist=min_dist,
                                           spread=spread,
                                           n_components=3,
                                           metric=metric,
                                           n_jobs=-1,  # Use all available cores
                                           random_state=42)

                    # Fit and transform the data
                    umap_results = umap_model.fit_transform(data_scaled)

                    # Create a new DataFrame for UMAP results
                    umap_df = pd.DataFrame(umap_results, columns=['UMAP-1', 'UMAP-2', 'UMAP-3'])
                    umap_df['Well_ID'] = data['Well_ID'].astype(str)
                    umap_df['Identifier'] = data['Identifier']

                    # Calculate mean coordinates for each Well_ID
                    mean_coords = umap_df.groupby('Well_ID')[['UMAP-1', 'UMAP-2', 'UMAP-3']].mean().reset_index()

                    # Create a color mapping for each unique Well_ID
                    unique_well_ids = mean_coords['Well_ID'].unique()
                    color_map = {well_id: color_palette[i % len(color_palette)] for i, well_id in enumerate(unique_well_ids)}

                    # Plot using Plotly
                    fig = px.scatter_3d(umap_df, x='UMAP-1', y='UMAP-2', z='UMAP-3', color='Well_ID', hover_data=['Identifier'], color_discrete_map=color_map)
                    fig.update_layout(title=f'{dataset_name} - 3D UMAP Projection with {metric}, n_neighbors={n_neighbors}, min_dist={min_dist}, spread={spread}')

                    # Add the mean coordinates to the plot with specific color and name
                    for _, row in mean_coords.iterrows():
                        well_id = row['Well_ID']
                        fig.add_trace(go.Scatter3d(x=[row['UMAP-1']], y=[row['UMAP-2']], z=[row['UMAP-3']],
                                                   mode='markers',
                                                   marker=dict(size=2, color=color_map[well_id]),
                                                   name=f'Mean of {well_id}'))

                    # Save Plotly fig as an HTML file for each dataset
                    fig.write_html(f'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/OUTPUT/umap_{dataset_name}_{metric}_n_neighbors_{n_neighbors}_min_dist_{min_dist}_spread_{spread}.html')
