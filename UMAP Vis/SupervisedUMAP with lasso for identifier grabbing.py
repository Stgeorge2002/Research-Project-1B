import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import matplotlib.colors as mcolors

# Ensure all required packages are installed
def ensure_packages_installed():
    required_packages = ["umap-learn", "pandas", "numpy", "matplotlib", "scikit-learn"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

ensure_packages_installed()

# Load your dataset
def load_data(filepath):
    return pd.read_csv(filepath)

# Convert categorical labels to numerical labels
def encode_labels(labels):
    label_codes, uniques = pd.factorize(labels)
    return label_codes, dict(enumerate(uniques))

# UMAP analysis function
def run_umap(data, target, n_neighbors=9, min_dist=0.001, n_components=2, metric='chebyshev'):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
        random_state=42, metric=metric, target_weight=0.4
    )
    embedding = reducer.fit_transform(data, y=target)
    return embedding

# Function to generate distinct colors
def generate_distinct_colors(num_colors):
    hsv_colors = [(x / num_colors, 0.85, 0.85) for x in range(num_colors)]
    return [mcolors.hsv_to_rgb(color) for color in hsv_colors]

# Main script
if __name__ == "__main__":
    filepath = 'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/trimmed_dataset.csv'
    data = load_data(filepath)
    data_for_umap = data.iloc[:, 9:]  # Assuming features start from the 10th column
    identifiers = data['Identifier']
    treatments = data['Treatment (1 d)']
    combined_labels = treatments.astype(str) + ' - ' + data['Coating (7 d)'].astype(str)

    treatment_codes, treatment_map = encode_labels(treatments)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_umap)

    umap_results = run_umap(data_scaled, treatment_codes)

    umap_df = pd.DataFrame(umap_results, columns=['UMAP-1', 'UMAP-2'])
    umap_df['Identifier'] = identifiers
    umap_df['Treatment (1 d)'] = treatments
    umap_df['Combined_Label'] = combined_labels

    unique_treatments = combined_labels.unique()
    colors = generate_distinct_colors(len(unique_treatments))
    color_map = {label: color for label, color in zip(unique_treatments, colors)}

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter_colors = [color_map[label] for label in combined_labels]
    scatter = ax.scatter(umap_df['UMAP-1'], umap_df['UMAP-2'], c=scatter_colors, s=20, alpha=0.2)
    ax.set_title('2D UMAP Projection')
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')

    legend_handles = [Patch(color=color_map[label], label=label) for label in unique_treatments]
    ax.legend(handles=legend_handles, title="Treatments")

    def onselect(verts):
        path = Path(verts)
        ind = np.nonzero(path.contains_points(scatter.get_offsets()))[0]
        selected_identifiers = umap_df.iloc[ind]['Identifier'].tolist()
        with open('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/OUTPUT2D/selected_identifiers.txt', 'w') as f:
            for identifier in selected_identifiers:
                f.write("%s\n" % identifier)
        print(f"Selected {len(selected_identifiers)} data points. Identifiers saved to 'selected_identifiers.txt'.")

    lasso = LassoSelector(ax, onselect, useblit=True)
    plt.show()
