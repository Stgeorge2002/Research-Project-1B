import pandas as pd

def load_datasets(original_path, filtered_path):
    """Load the original and filtered datasets from CSV files."""
    original_df = pd.read_csv(original_path)
    filtered_df = pd.read_csv(filtered_path)
    return original_df, filtered_df

def subtract_datasets(original_df, filtered_df):
    """Subtract the filtered dataset from the original dataset, starting from the 6th column."""
    # Identify numeric columns present in both datasets
    common_numeric_columns = original_df.iloc[:, 5:].select_dtypes(include='number').columns.intersection(
        filtered_df.columns)

    inverse_df = original_df.copy()
    inverse_df[common_numeric_columns] = original_df[common_numeric_columns] - filtered_df[common_numeric_columns]
    return inverse_df

def create_new_column_names(original_df, filtered_df):
    """Create new column names based on the original and filtered dataset column names."""
    new_columns = original_df.columns.tolist()
    for i, col in enumerate(original_df.columns[5:], start=5):
        new_columns[i] = f"{col}-minus-{filtered_df.columns.get(i, 'NA')}"
    return new_columns

def save_dataset(df, path):
    """Save the dataset to a CSV file."""
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}")

def main():
    
    # Paths to the datasets
    # Paths to the datasets
    original_dataset_path = 'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/test/AD.csv'
    filtered_dataset_path = 'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/test/filtered_dataset.csv'
    output_dataset_path = 'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/test'

    # Load datasets
    original_df, filtered_df = load_datasets(original_dataset_path, filtered_dataset_path)

    # Subtract datasets
    inverse_df = subtract_datasets(original_df, filtered_df)

    # Create new column names
    new_column_names = create_new_column_names(original_df, filtered_df)
    inverse_df.columns = new_column_names

    # Save the inverse dataset
    save_dataset(inverse_df, output_dataset_path)

if __name__ == "__main__":
    main()
