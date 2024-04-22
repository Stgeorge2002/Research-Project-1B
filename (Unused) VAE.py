# Install required packages
import subprocess
import sys

def install_packages():
    packages = ['numpy', 'pandas', 'tensorflow', 'scikit-learn', 'matplotlib']
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from sklearn.preprocessing import RobustScaler, LabelEncoder
    from sklearn.metrics import silhouette_samples
    from scipy.cluster.hierarchy import linkage, fcluster
except ImportError:
    install_packages()  # Install if not already installed
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from sklearn.preprocessing import RobustScaler, LabelEncoder
    from sklearn.metrics import silhouette_samples
    from scipy.cluster.hierarchy import linkage, fcluster

# Define Sampling layer for VAE
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Function to build and compile the VAE
def build_vae(input_dim, latent_dim):
    # Encoder
    encoder_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(encoder_inputs)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    
    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation='relu')(latent_inputs)
    decoder_outputs = layers.Dense(input_dim, activation='sigmoid')(x)
    decoder = Model(latent_inputs, decoder_outputs, name='decoder')
    
    # VAE
    class VAE(Model):
        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder

        def call(self, inputs):
            z_mean, z_log_var, z = self.encoder(inputs)
            reconstructed = self.decoder(z)
            kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
            self.add_loss(kl_loss)
            return reconstructed
    
    vae = VAE(encoder, decoder)
    vae.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return vae, encoder, decoder

# Main function to process the dataset
def process_dataset(file_path, latent_dim=10, sample_target=500):
    # Load dataset
    df = pd.read_csv(file_path)
    X_categorical = df.iloc[:, :5]  # Adjust based on your dataset
    X_morphogenic = df.iloc[:, 9:]  # Adjust based on your dataset
    y = df['Treatment (1 d)']  # Adjust based on your dataset

    # Data preprocessing
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_morphogenic)

    # Build and train VAE
    vae, vae_encoder, _ = build_vae(X_scaled.shape[1], latent_dim)
    vae.fit(X_scaled, X_scaled, epochs=30, batch_size=32)

    # Transform data into latent space
    z_mean, _, _ = vae_encoder.predict(X_scaled)

    # Sample selection based on silhouette scores and hierarchical clustering
    selected_samples = []
    for label in np.unique(y_encoded):
        label_indices = np.where(y_encoded == label)[0]
        label_z = z_mean[label_indices]

        # Hierarchical clustering
        Z = linkage(label_z, 'ward')
        max_d = 7.0  # Adjust based on your data
        clusters = fcluster(Z, max_d, criterion='distance')
        
        # Silhouette analysis
        silhouette_vals = silhouette_samples(label_z, clusters)
        selected_idx = np.argsort(-silhouette_vals)[:sample_target // len(np.unique(y_encoded))]
        selected_samples.extend(label_indices[selected_idx])

    # Compile selected data
    X_final = X_scaled[selected_samples, :]
    y_final = y_encoded[selected_samples]
    df_final = pd.DataFrame(X_final, columns=df.columns[9:])  # Adjust based on your dataset
    df_final['M_Type'] = encoder.inverse_transform(y_final)  # Add decoded M types
    df_final.to_csv('/path/to/final_selected_samples.csv', index=False)
    print("Selected samples and features saved to final_selected_samples.csv")

# Run the script with your dataset
if __name__ == "__main__":
    process_dataset('/path/to/your/Mnone.csv')
