import pandas as pd
import umap
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/AD.csv')
data_for_umap = data.iloc[:, 5:].dropna()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_umap)

# Define the metrics and parameters to iterate over
metrics = ["euclidean", "manhattan", "chebyshev", "minkowski", "canberra", "braycurtis", "haversine", "mahalanobis", "wminkowski", "seuclidean"]
n_neighbors_options = [5, 10, 15]  # Example ranges
min_dist_options = [0.1, 0.5, 0.99]  # Example ranges

for metric in metrics:
    for n_neighbors in n_neighbors_options:
        for min_dist in min_dist_options:
            # Initialize UMAP
            umap_model = umap.UMAP(n_neighbors=n_neighbors,
                                   min_dist=min_dist,
                                   n_components=3,
                                   metric=metric,
                                   random_state=42)

            # Fit and transform the data
            umap_results = umap_model.fit_transform(data_scaled)

            # Create a new DataFrame for UMAP results
            umap_df = pd.DataFrame(umap_results, columns=['UMAP-1', 'UMAP-2', 'UMAP-3'])
            umap_df['Well_ID'] = data['Well_ID']
            umap_df['Identifier'] = data['Identifier']

            # Plot
            fig = px.scatter_3d(umap_df, x='UMAP-1', y='UMAP-2', z='UMAP-3', color='Well_ID', hover_data=['Identifier'])
            fig.update_layout(title=f'3D UMAP Projection with {metric}, n_neighbors={n_neighbors}, min_dist={min_dist}')
            fig.show()

