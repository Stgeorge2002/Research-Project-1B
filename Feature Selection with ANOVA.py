import pandas as pd
from scipy.stats import f_oneway

# Load your data into a pandas DataFrame
data = pd.read_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/DATA/ADB.csv')  # Adjust this to the path of your CSV file

# Exclude the first seven columns and the 'ID' column for the ANOVA test
features_data = data.iloc[:, 7:]  # Assuming the 'ID' is the 7th column, adjust if needed
grouping_variable = data['ID']  # Adjust if the 'ID' column has a different name

# Prepare a dictionary to store the ANOVA results
anova_results = {}

# Perform the ANOVA test for each feature
for feature in features_data.columns:
    groups = [group[1] for group in features_data.groupby(grouping_variable)[feature]]
    f_stat, p_value = f_oneway(*groups)
    anova_results[feature] = (f_stat, p_value)

# Output the ANOVA results
print("ANOVA Results (F-statistic and P-value) for Each Feature:")
for feature, (f_stat, p_value) in anova_results.items():
    print(f"{feature}: F-statistic = {f_stat}, P-value = {p_value}")

# Convert the ANOVA results to a DataFrame
anova_results_df = pd.DataFrame.from_dict(anova_results, orient='index', columns=['F_statistic', 'P_value'])

# Save the DataFrame to a CSV file
csv_file_path = 'C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/StatOut/anovanew.csv'
anova_results_df.to_csv(csv_file_path, index_label='Feature')
