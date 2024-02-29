import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Attempt to import necessary packages, install if they're missing
try:
    import umap
except ImportError:
    install("umap-learn")
    import umap

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    install("scikit-learn")
    from sklearn.preprocessing import StandardScaler

try:
    import plotly.express as px
except ImportError:
    install("plotly")
    import plotly.express as px

import pandas as pd

# Load the dataset
data = pd.read_csv(r'C:\Users\theoa\OneDrive\Desktop\Bath\Research Project 1B\AD.csv')

# Exclude the first five columns and drop rows with missing values
data_for_umap = data.iloc[:, 5:].dropna()

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_for_umap)

# Initialize UMAP
umap_model = umap.UMAP(n_neighbors=10,  # Adjust based on dataset
                       min_dist=0.1,   # Adjust based on dataset
                       n_components=2,
                       random_state=42,
                       metric='canberra')

# Fit the model and transform the data
umap_results = umap_model.fit_transform(data_scaled)

# Add UMAP results to the original dataframe
data['UMAP-1'] = umap_results[:, 0]
data['UMAP-2'] = umap_results[:, 1]

# Interactive plot using Plotly
fig = px.scatter(data, x='UMAP-1', y='UMAP-2', color='Well_ID', hover_data=['Identifier'])
fig.update_layout(title='UMAP Projection of AD Dataset', width=800, height=600)
fig.show()
