import pandas as pd

# Load the dataset
data_df = pd.read_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/Macs.csv')

# Define rows and columns to filter based on the image
rows_to_filter_treat_sis = [7]  # Row 7 for Treat-SIS
rows_to_filter_coat_sis = [12, 13, 14]  # Rows 12, 13, 14 for Coat-SIS
columns_to_filter = [3, 4, 5, 6, 7, 8, 9, 10, 11]  # Columns to filter

# Since pandas uses zero-based indexing, subtracting 1 from row and column numbers
rows_to_filter_treat_sis = [x - 1 for x in rows_to_filter_treat_sis]
rows_to_filter_coat_sis = [x - 1 for x in rows_to_filter_coat_sis]
columns_to_filter = [x - 1 for x in columns_to_filter]

# Filter the dataset for Treat-SIS and Coat-SIS
treat_sis_df = data_df.iloc[rows_to_filter_treat_sis, columns_to_filter]
coat_sis_df = data_df.iloc[rows_to_filter_coat_sis, columns_to_filter]

# Add a new column for naming the rows appropriately
treat_sis_df.insert(0, 'Name', ['Treat-SIS'] * len(rows_to_filter_treat_sis))
coat_sis_df.insert(0, 'Name', ['Coat-SIS'] * len(rows_to_filter_coat_sis))

# Combine both DataFrames
combined_df = pd.concat([treat_sis_df, coat_sis_df])

# Export to CSV
combined_df.to_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/iltered_data.csv', index=False)
