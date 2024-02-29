import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Load your data
data = pd.read_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/ADB.csv')  # Adjust this to the path of your CSV file

# Extract features from the 7th column onwards
features_data = data.iloc[:, 6:]

# Number of times to split and compare
n_splits = 100  # Adjust as needed for your analysis

# Initialize a dictionary to store the cumulative K-S statistics for each feature
cumulative_ks_statistics = {feature: 0 for feature in features_data.columns}

for _ in range(n_splits):
    # Randomly split the data into two halves
    shuffled_indices = np.random.permutation(features_data.index)
    split_index = len(shuffled_indices) // 2
    indices1, indices2 = shuffled_indices[:split_index], shuffled_indices[split_index:]
    data1, data2 = features_data.loc[indices1], features_data.loc[indices2]
    
    # Compare the distributions of each feature using the K-S test and accumulate the statistics
    for feature in features_data.columns:
        ks_stat, _ = ks_2samp(data1[feature], data2[feature])
        cumulative_ks_statistics[feature] += ks_stat

# Compute the average K-S statistic for each feature
average_ks_statistics = {feature: ks_stat / n_splits for feature, ks_stat in cumulative_ks_statistics.items()}

# Output the average K-S statistics
print("Average K-S Statistics for Each Feature:")
for feature, avg_ks_stat in average_ks_statistics.items():
    print(f"{feature}: {avg_ks_stat}")

# Convert the average K-S statistics to a DataFrame
average_ks_stats_df = pd.DataFrame(list(average_ks_statistics.items()), columns=['Feature', 'Average_KS_Statistic'])

# Save the DataFrame to a CSV file
csv_file_path = 'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/StatOut/average_ks_statistics.csv'
average_ks_stats_df.to_csv(csv_file_path, index=False)

csv_file_path