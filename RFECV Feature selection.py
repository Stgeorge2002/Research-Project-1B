import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages that your script depends on
required_packages = [
    'numpy',
    'pandas',
    'scikit-learn',
    'scipy',
    'xgboost'
]

for package in required_packages:
    try:
        install(package)
    except Exception as e:
        print(f"An error occurred while installing {package}: {e}", file=sys.stderr)

# Now import the installed packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import linkage, fcluster
from xgboost import XGBClassifier

# Continue with the rest of your script...
df = pd.read_csv('/home/1Llight/Mnone.csv')
# The rest of your data processing and machine learning code follows...


# Separate data
X_categorical = df.iloc[:, :9]  # Categorical data
X_morphogenic = df.iloc[:, 9:]  # Morphogenic features
y = X_categorical['Treatment (1 d)']  # Assuming 'Treatment' column exists

# Encode categorical data
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Scale morphogenic features
scaler = RobustScaler()
X_morphogenic_scaled = scaler.fit_transform(X_morphogenic)

# Advanced Sample Selection
samples_per_type = 300  # Desired number of samples per M type
selected_indices = []
for label in np.unique(y_encoded):
    indices = np.where(y_encoded == label)[0]
    X_subset = X_morphogenic_scaled[indices, :]
    
    # Hierarchical clustering
    Z = linkage(X_subset, 'ward')
    clusters = fcluster(Z, t=3, criterion='maxclust')  # Aim for a reasonable number of clusters
    
    # Select samples from clusters, aiming for even distribution
    cluster_indices = np.array([indices[clusters == i] for i in np.unique(clusters)])
    num_clusters = len(cluster_indices)
    
    samples_per_cluster = samples_per_type // num_clusters
    remainder = samples_per_type % num_clusters
    for i, cluster in enumerate(cluster_indices):
        selected_num = samples_per_cluster + (1 if i < remainder else 0)
        if len(cluster) > selected_num:
            silhouette_vals = silhouette_samples(X_morphogenic_scaled[cluster, :], clusters[clusters == np.unique(clusters)[i]])
            top_silhouette_indices = np.argsort(-silhouette_vals)[:selected_num]
            selected_indices.extend(cluster[top_silhouette_indices])
        else:
            selected_indices.extend(cluster)

# Enhanced Feature Selection with RFECV
estimator = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
selector = RFECV(estimator, step=1, cv=StratifiedKFold(5), scoring='accuracy')
selector.fit(X_morphogenic_scaled[selected_indices, :], y_encoded[selected_indices])

X_selected_features = selector.transform(X_morphogenic_scaled[selected_indices, :])

# Combine selected features with original categorical data
X_final = np.hstack([X_categorical.values[selected_indices, :], X_selected_features])

# Export new dataset
feature_names = [f'Feature_{i}' for i in range(X_selected_features.shape[1])]
selected_df = pd.DataFrame(X_final, columns=list(X_categorical.columns) + feature_names)
selected_df.to_csv('/home/1Llight/Mnonei.csv', index=False)
