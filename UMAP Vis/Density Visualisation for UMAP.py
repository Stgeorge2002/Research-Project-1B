
import subprocess
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import plotly.express as px

# Function to install required packages
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
datasets = ['C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/trimmed_dataset.csv']
dataset_names = ['trimmed_dataset']

# Define the metrics and parameters to iterate over
metrics = ["chebyshev"]
n_neighbors_options = [9]
min_dist_options = [0.001]  # Ensure this is a float
spread_options = [1]  # Ensure this is a float
target_weight_options = [0.4]  # Ensure this is a float

# Perform UMAP for each combination of metrics and parameters
for dataset, dataset_name in zip(datasets, dataset_names):
    # Load the dataset
    data = pd.read_csv(dataset)

    # Replace NaN values with zero (assuming features start from the 10th column)
    data_for_umap = data.iloc[:, 9:].fillna(0)

    # Create combined labels dynamically
    combined_labels = data['Treatment (1 d)'].astype(str) + ' - ' + data['Coating (7 d)'].astype(str)
    combined_labels = combined_labels.astype('category').cat.codes
    
    # Scaling the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_umap)

    for metric in metrics:
        for n_neighbors in n_neighbors_options:
            for min_dist in min_dist_options:
                for spread in spread_options:
                    for target_weight in target_weight_options:
                        # Initialize UMAP for 2D with supervised dimension reduction
                        umap_model = umap.UMAP(n_neighbors=n_neighbors,
                                               min_dist=min_dist,
                                               spread=spread,
                                               n_components=2, 
                                               metric=metric,
                                               n_jobs=1,  
                                               random_state=42,
                                               target_weight=target_weight)

                        # Fit and transform the data using combined labels
                        umap_results = umap_model.fit_transform(data_scaled, y=combined_labels)

                        # Create a new DataFrame for UMAP results
                        umap_df = pd.DataFrame(umap_results, columns=['UMAP-1', 'UMAP-2'])
                        umap_df['Combined_Label'] = data['Treatment (1 d)'].astype(str) + ' - ' + data['Coating (7 d)'].astype(str)

                        # Create a color mapping for each unique combined label
                        unique_labels = umap_df['Combined_Label'].unique()
                        color_palette = px.colors.qualitative.Plotly
                        color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels)}

                        # Create a density contour plot using Plotly for 2D
                        fig = px.density_contour(umap_df, x='UMAP-1', y='UMAP-2', color='Combined_Label', color_discrete_map=color_map)
                        
                        fig.update_layout(title=f'{dataset_name} - 2D UMAP Density Plot with {metric}', plot_bgcolor='white')
                        
                        # Save Plotly fig as an HTML file
                        fig.write_html(f'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/{dataset_name}_2d_umap_density_plot.html')
