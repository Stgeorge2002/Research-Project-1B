import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Load your dataset
dataset_path = 'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/Mnone.csv'  # Update this with your actual dataset path
df = pd.read_csv(dataset_path)

# Separate the dataset into morphogenic features and treatment labels
X_morphogenic = df.iloc[:, 9:]  # Morphogenic features
y = df['Treatment (1 d)']  # The treatment column

# Normalize morphogenic features
scaler = StandardScaler()
X_morphogenic_scaled = scaler.fit_transform(X_morphogenic)

# Define the autoencoder model
input_dim = X_morphogenic_scaled.shape[1]
encoding_dim = 32  # Adjust based on your needs

input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.2)(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(encoded)
decoded = BatchNormalization()(decoded)
decoded = Dropout(0.2)(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Split and train the autoencoder
X_train, X_test = train_test_split(X_morphogenic_scaled, test_size=0.2, random_state=42)
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# Encode the morphogenic data
X_encoded = encoder.predict(X_morphogenic_scaled)

# Cluster the encoded features and select representative samples
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_encoded)

selected_indices = []
for treatment in ['M0', 'M1', 'M2']:
    treatment_indices = (y == treatment)
    for cluster in range(3):  # Assuming three clusters
        in_cluster = clusters == cluster
        indices = np.where(treatment_indices & in_cluster)[0]
        if len(indices) > 0:
            distances = cdist(X_encoded[indices], [kmeans.cluster_centers_[cluster]])
            num_samples = 50  # Number of samples per treatment and cluster
            selected_indices.extend(indices[np.argsort(distances, axis=0)[:num_samples].flatten()])


# Feature Selection using Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_morphogenic_scaled, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Select the top N features
N = 30  # Number of top features to select, adjust this based on your needs
X_top_features = X_morphogenic.iloc[:, indices[:N]]

# Create the final dataset with selected samples and top features
selected_samples_morphogenic = X_top_features.iloc[selected_indices]
selected_samples_categorical = df.iloc[selected_indices, :9]  # Categorical data
final_dataset = pd.concat([selected_samples_categorical.reset_index(drop=True), selected_samples_morphogenic.reset_index(drop=True)], axis=1)

# Save the final dataset
final_dataset.to_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/MnoneEncode50.csv', index=False)
