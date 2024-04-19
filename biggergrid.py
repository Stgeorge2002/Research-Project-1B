import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

# Load the dataset
def load_data(filepath):
    return pd.read_csv(filepath)

# Define UMAP function
def run_umap(data, target, n_neighbors=9, min_dist=0.001, n_components=2, metric='chebyshev', target_weight=0.4):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric,
                        target_weight=target_weight, random_state=42)
    return reducer.fit_transform(data, y=target)

# Main execution
if __name__ == "__main__":
    filepath = 'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/trimmed_dataset.csv'
    data = load_data(filepath)
    data_for_umap = data.iloc[:, 9:].fillna(0)  # Replace NaNs with zeros
    feature_names = data_for_umap.columns  # Collect feature names

    # Create combined labels dynamically for supervision
    combined_labels = data['Treatment (1 d)'].astype(str) + ' - ' + data['Coating (7 d)'].astype(str)
    combined_labels = combined_labels.astype('category').cat.codes

    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_umap)

    # Generate UMAP embedding with supervision
    umap_results = run_umap(data_scaled, combined_labels)

    # Determine number of features and grid size
    num_features = data_for_umap.shape[1]
    num_plots_side = int(np.ceil(np.sqrt(num_features)))  # Calculate grid size

    # Create matplotlib figure and axes
    fig, axes = plt.subplots(nrows=num_plots_side, ncols=num_plots_side, figsize=(15, 15))
    axes = axes.flatten()

    # Plot each feature
    for i in range(num_plots_side * num_plots_side):
        ax = axes[i]
        if i < num_features:
            sc = ax.scatter(umap_results[:, 0], umap_results[:, 1], c=data_scaled[:, i], cmap='viridis', s=5, alpha=0.6)
            ax.set_title(feature_names[i], fontsize=8)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')  # Turn off unused subplots

    plt.tight_layout()
    plt.show()
