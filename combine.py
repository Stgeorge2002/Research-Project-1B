import pandas as pd

# Function to load datasets
def load_datasets(filepath1, filepath2):
    data1 = pd.read_csv(filepath1)
    data2 = pd.read_csv(filepath2)
    return data1, data2

# Function to trim the second dataset to match selected features from the first
def trim_features(data1, data2, starting_col=9):
    # Assuming the first 9 columns are identical and should be ignored in the comparison
    selected_features = data1.columns[starting_col:]
    shared_features = [feature for feature in data2.columns[starting_col:] if feature in selected_features]
    # Create a list of columns to keep in data2 (first 9 columns + shared features)
    columns_to_keep = list(data2.columns[:starting_col]) + shared_features
    trimmed_data2 = data2[columns_to_keep]
    return trimmed_data2

# Main function to execute the process
def process_datasets(filepath1, filepath2):
    # Load data
    data1, data2 = load_datasets(filepath1, filepath2)
    # Trim the second dataset
    trimmed_data2 = trim_features(data1, data2)
    # Save the trimmed dataset or return it
    trimmed_data2.to_csv('trimmed_dataset.csv', index=False)
    return trimmed_data2

# Example of how to call the function
# This call is commented out because function calls should not be active when sharing code
# Replace 'dataset1.csv' and 'dataset2.csv' with your actual file paths
trimmed_data = process_datasets('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/lime and sharp/more_cleaned_mnone.csv', 'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/finer data/New Data.csv')
