import sys
import subprocess
import pkg_resources

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check if umap-learn is installed, and if not, install it
required = {'umap-learn'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    for package in missing:
        install(package)

# Now, import the umap package
import umap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r'C:\Users\theoa\OneDrive\Desktop\Bath\Research Project 1B\AD.csv')

# Exclude the first five columns
data_for_umap = data.iloc[:, 5:]

# Initialize UMAP
umap_model = umap.UMAP(n_neighbors=100, min_dist=0.01, n_components=2, random_state=42)

# Fit the model and transform the data
umap_results = umap_model.fit_transform(data_for_umap)

# Convert the results into a DataFrame for easy plotting
umap_df = pd.DataFrame(umap_results, columns=['UMAP-1', 'UMAP-2'])

# Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(x='UMAP-1', y='UMAP-2', data=umap_df)
plt.title('UMAP Projection')
plt.show()
