import pandas as pd

# Load the datasets
data_df = pd.read_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/finer data/Data.csv')  # Replace 'path_to_your_Data.csv' with the actual path
key_df = pd.read_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/finer data/Key.csv')    # Replace 'path_to_your_Key.csv' with the actual path

# Clean up the column names in the Data dataset
data_df.columns = [col.strip() for col in data_df.columns]

# Merging the 'data_df' with 'key_df' based on 'Row', 'Column', and 'Experiment' columns
merged_df = pd.merge(data_df, key_df, on=['Row', 'Column', 'Experiment'], how='left')

# Reordering the columns to place 'Coating (7 d)' and 'Treatment (1 d)' right after 'Identifier'
columns_order = ['Experiment', 'ID', 'Identifier', 'Coating (7 d)', 'Treatment (1 d)'] + \
                [col for col in merged_df.columns if col not in ['Experiment', 'ID', 'Identifier', 'Coating (7 d)', 'Treatment (1 d)']]
merged_df = merged_df[columns_order]

# Save the updated dataframe to a new CSV file
merged_df.to_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/finer data/New Data.csv', index=False)
