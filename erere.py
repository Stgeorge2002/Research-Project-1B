import subprocess
import sys
import pandas as pd
import umap
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

# Load the dataset
data = pd.read_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/test/AD_modified.csv')

# Select the relevant columns and replace NaN values with zero
data_for_umap = data.iloc[:, 5:]

# Scaling the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_umap)

# Define the metrics and parameters to iterate over
metrics = ["manhattan"]
n_neighbors_options = [70, 100] # Example ranges for n_neighbors
min_dist_options = [0.]  # Example ranges for min_dist
spread_options = [1, 3] # Example ranges for spread

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
                                       random_state=42)  # Comment out for parallel processing

                # Fit and transform the data
                umap_results = umap_model.fit_transform(data_scaled)

                # Extract the UMAP graph
                umap_graph = umap_model.graph_

                # Create a 3D scatter plot for the UMAP points
                fig = go.Figure(data=[go.Scatter3d(
                    x=umap_results[:, 0], y=umap_results[:, 1], z=umap_results[:, 2],
                    mode='markers', 
                    marker=dict(size=2, color='blue'))])

                # Update layout of the figure
                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

                # Save Plotly fig as an HTML file
                fig.write_html(f'umap_{metric}_n_neighbors_{n_neighbors}_min_dist_{min_dist}_spread_{spread}.html')
