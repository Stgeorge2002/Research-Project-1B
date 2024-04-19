TF_ENABLE_ONEDNN_OPTS=0

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages that your script depends on
required_packages = [
    'numpy',
    'pandas',
    'scikit-learn',
    'umap-learn',
    'hdbscan',
    'matplotlib'
]

for package in required_packages:
    try:
        install(package)
    except Exception as e:
        print(f"An error occurred while installing {package}: {e}")

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
import umap
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/Mnone.csv')

# Separate the data into morphogenic features and treatment labels
X_morphogenic = df.iloc[:, 9:].values  # Adjust if necessary based on your dataset structure
y = df['Treatment (1 d)'].values  # Treatment labels

# Normalize morphogenic features using robust scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_morphogenic)

# Generate polynomial and interaction features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Select features based on mutual information and keep top 10%
mi = mutual_info_classif(X_poly, y)
threshold = np.percentile(mi, 90)
high_mi_features = mi >= threshold
X_high_mi = X_poly[:, high_mi_features]

# Further reduce dimensions using Recursive Feature Elimination with Cross-Validation (RFECV)
selector = RFECV(estimator=RandomForestClassifier(n_estimators=100, random_state=42), step=0.1, cv=5, scoring='accuracy')
X_selected = selector.fit_transform(X_high_mi, y)

# Optimize UMAP parameters using silhouette score for better clustering differentiation
best_silhouette = -1
best_umap = None
best_params = {}
n_neighbors_options = [5, 15, 30, 50]
min_dist_options = [0.0, 0.1, 0.3, 0.5]
metric_options = ['euclidean', 'cosine', 'manhattan', 'chebyshev', 'minkowski']

for n_neighbors in n_neighbors_options:
    for min_dist in min_dist_options:
        for metric in metric_options:
            umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, n_components=2, random_state=42)
            X_umap = umap_reducer.fit_transform(X_selected)
            
            # Clustering on reduced data
            clusterer = HDBSCAN(min_samples=10, min_cluster_size=20)
            labels = clusterer.fit_predict(X_umap)
            
            # Evaluate clusters if more than one cluster is found
            if len(set(labels)) - (1 if -1 in labels else 0) > 1:  # Check if more than one cluster excluding noise
                silhouette_avg = silhouette_score(X_umap, labels)
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_umap = X_umap
                    best_params = {'n_neighbors': n_neighbors, 'min_dist': min_dist, 'metric': metric}

# Apply best UMAP parameters to the entire dataset
final_umap_reducer = umap.UMAP(n_neighbors=best_params['n_neighbors'], min_dist=best_params['min_dist'], metric=best_params['metric'], n_components=2, random_state=42)
X_umap_final = final_umap_reducer.fit_transform(X_selected)

# Final clustering using HDBSCAN
final_clusterer = HDBSCAN(min_samples=10, min_cluster_size=20)
final_labels = final_clusterer.fit_predict(X_umap_final)

# Visualize the final clustering
plt.figure(figsize=(12, 10))
plt.scatter(X_umap_final[:, 0], X_umap_final[:, 1], c=final_labels, cmap='Spectral', s=50, alpha=0.6)
plt.title('Optimized UMAP with HDBSCAN Clustering')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar()
plt.show()
