# Full Python script for feature selection based on Spearman correlation and bootstrapping

# Import necessary libraries
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Load the dataset
data = pd.read_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/Mnone.csv')  # Update the path to your dataset

# Separate numeric and categorical features
X = data.iloc[:, 9:]  # Assuming numeric features start from the 10th column
categorical_data = data.iloc[:, :9]  # Assuming first 9 columns are categorical

# Standardize the numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to compute Spearman correlation matrix efficiently
def compute_spearman_matrix(X):
    ranks = np.apply_along_axis(rankdata, 0, X)
    n = X.shape[1]
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            corr, _ = spearmanr(ranks[:, i], ranks[:, j])
            corr_matrix[i, j] = corr_matrix[j, i] = corr
    return corr_matrix

# Compute the initial Spearman correlation matrix
corr_matrix = compute_spearman_matrix(X_scaled)

# Convert correlation to dissimilarity
dissimilarity_matrix = 1 - np.abs(corr_matrix)

# Perform bootstrapping for variable importance
n_bootstraps = 100  # Adjust based on computational capability
variable_importance = np.zeros(X.shape[1])
for _ in range(n_bootstraps):
    indices = np.random.randint(0, X_scaled.shape[0], X_scaled.shape[0])
    X_resampled = X_scaled[indices]
    resampled_corr = compute_spearman_matrix(X_resampled)
    resampled_dissimilarity = 1 - np.abs(resampled_corr)
    variable_importance += np.sum(np.abs(dissimilarity_matrix - resampled_dissimilarity), axis=0)
variable_importance /= (n_bootstraps * X.shape[1])

# Select variables based on importance scores
threshold = np.mean(variable_importance)  # You can choose a different threshold
selected_features = X.columns[variable_importance > threshold]

# Prepare the final dataset including selected features and all categorical columns
final_dataset = pd.concat([categorical_data, X.loc[:, selected_features]], axis=1)

# Save the final dataset
final_dataset_path = 'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/Mnone_selected_features.csv'  # Update the path as necessary
final_dataset.to_csv(final_dataset_path, index=False)
