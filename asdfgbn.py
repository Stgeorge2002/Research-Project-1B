import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Attempt to import umap and sklearn, install if they're missing
try:
    import umap
except ImportError:
    install("umap-learn")
    import umap

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    install("scikit-learn")
    from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r'C:\Users\theoa\OneDrive\Desktop\Bath\Research Project 1B\AD.csv')

# Exclude the first five columns and drop rows with missing values
data_for_umap = data.iloc[:, 5:].dropna()

# Apply Min-Max scaling
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_for_umap)

# Initialize UMAP
umap_model = umap.UMAP(n_neighbors=30,  # Adjust based on dataset
                       min_dist=0.0,   # Adjust based on dataset
                       n_components=2,
                       random_state=42)

# Fit the model and transform the data
umap_results = umap_model.fit_transform(data_scaled)

# Print the shape of the UMAP embedding
print("Shape of UMAP embedding:", umap_results.shape)

# Convert the results into a DataFrame for easy plotting
umap_df = pd.DataFrame(umap_results, columns=['UMAP-1', 'UMAP-2'])

# Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(x='UMAP-1', y='UMAP-2', data=umap_df)
plt.title('UMAP Projection of AD Dataset')
plt.show()
