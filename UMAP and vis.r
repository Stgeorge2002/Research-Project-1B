import subprocess
import sys

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

import pandas as pd
import umap
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/test/AD.csv')
data_for_umap = data.iloc[:, 5:].dropna()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_umap)

# Define the metrics and parameters to iterate over
metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]
n_neighbors_options = [5, 20, 40]  # Example ranges
min_dist_options = [0.1, 0.2, 0.5]  # Example ranges

for metric in metrics:
    for n_neighbors in n_neighbors_options:
        for min_dist in min_dist_options:
            # Initialize UMAP
            umap_model = umap.UMAP(n_neighbors=n_neighbors,
                                   min_dist=min_dist,
                                   n_components=3,
                                   metric=metric,
                                   n_jobs=-1,  # Use all available cores
                                   random_state=42)  # Comment out for parallel processing

            # Fit and transform the data
            umap_results = umap_model.fit_transform(data_scaled)

            # Create a new DataFrame for UMAP results
            umap_df = pd.DataFrame(umap_results, columns=['UMAP-1', 'UMAP-2', 'UMAP-3'])
            umap_df['Well_ID'] = data['Well_ID'].astype(str)  # Ensure Well_ID is treated as categorical
            umap_df['Identifier'] = data['Identifier']

            # Calculate mean coordinates for each Well_ID
            mean_coords = umap_df.groupby('Well_ID')[['UMAP-1', 'UMAP-2', 'UMAP-3']].mean().reset_index()

            # Plot using Plotly
            fig = px.scatter_3d(umap_df, x='UMAP-1', y='UMAP-2', z='UMAP-3', color='Well_ID', hover_data=['Identifier'], color_discrete_sequence=px.colors.qualitative.Plotly)
            fig.update_layout(title=f'3D UMAP Projection with {metric}, n_neighbors={n_neighbors}, min_dist={min_dist}')

            # Add the mean coordinates to the plot
            for _, row in mean_coords.iterrows():
                fig.add_trace(px.scatter_3d(x=[row['UMAP-1']], y=[row['UMAP-2']], z=[row['UMAP-3']], 
                                             marker=dict(size=5, color='black')).data[0])

            # Save Plotly fig as an HTML file
            fig.write_html(f'umap_{metric}_n_neighbors_{n_neighbors}_min_dist_{min_dist}.html')
