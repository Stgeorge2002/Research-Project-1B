import subprocess
import sys

# Function to install required packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = ["umap-learn", "scikit-learn", "shap", "lime", "tensorflow", "keras", "numpy", "pandas"]

# Install missing packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# Import necessary libraries
import subprocess
import sys
import numpy as np
import pandas as pd
import umap
import shap
from lime import lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler

# Path to your dataset
dataset_path = 'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/Mnone.csv'  # Change this to the path of your dataset

# Column name of the labels or outcomes in your dataset
label_column = 'Treatment (1 d)'

# Load your dataset
df = pd.read_csv(dataset_path)

# Drop the first 9 categorical columns
X = df.drop(df.columns[0:9].tolist() + [label_column], axis=1).values
y = df[label_column].values

# Scale the features (important for models and UMAP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# UMAP parameters
metrics = ["chebyshev"]
n_neighbors_options = [8, 12]
min_dist_options = [0.0]

# Loop through UMAP configurations
for metric in metrics:
    for n_neighbors in n_neighbors_options:
        for min_dist in min_dist_options:
            # Apply UMAP
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
            X_reduced = reducer.fit_transform(X)

            # Split the reduced data
            X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

            # Fit a RandomForest model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Apply SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, show=False)
            shap_df = pd.DataFrame(shap_values[1], columns=['SHAP_Feature1', 'SHAP_Feature2'])
            shap_df.to_csv(f'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/shap_results_{metric}_n_{n_neighbors}_md_{min_dist}.csv', index=False)

            # Apply LIME
            explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=['UMAP Feature 1', 'UMAP Feature 2'], class_names=['Class 0', 'Class 1', 'Class 2'], discretize_continuous=True)
            lime_results = []
            for i in range(len(X_test)):
                exp = explainer.explain_instance(X_test[i], model.predict_proba, num_features=2, labels=(0, 1, 2))
                intercep, local_exp = exp.local_exp[1]
                lime_results.append(local_exp)
            lime_df = pd.DataFrame(lime_results, columns=['LIME_Feature1', 'LIME_Feature2'])
            lime_df.to_csv(f'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/lime_results_{metric}_n_{n_neighbors}_md_{min_dist}.csv', index=False)

            # Autoencoder for reconstruction error
            # Preprocess data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Define autoencoder architecture
            input_dim = X_reduced.shape[1]
            encoding_dim = 2
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(encoding_dim, activation='relu')(input_layer)
            decoded = Dense(input_dim, activation='sigmoid')(encoded)
            autoencoder = Model(input_layer, decoded)
            encoder = Model(input_layer, encoded)
            autoencoder.compile(optimizer='adam', loss='mse')

            # Train the autoencoder
            autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

            # Evaluate reconstruction error
            reconstructions = autoencoder.predict(X_test)
            mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
            mse_df = pd.DataFrame(mse, columns=['MSE'])
            mse_df.to_csv(f'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/autoencoder_recon_error_{metric}_n_{n_neighbors}_md_{min_dist}.csv', index=False)
