import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV

# Load your data
data = pd.read_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/test/AD.csv')  # Make sure to provide the correct file path

# Prepare the features and target variable
X = data.iloc[:, 6:-1]  # Features from the 7th column to the second last
y = data.iloc[:, -1]    # Assuming the last column as target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Elastic Net
elastic_net = ElasticNetCV(cv=5, random_state=0, max_iter=10000)
elastic_net.fit(X_train, y_train)

# Extract and display non-zero coefficients
features = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': elastic_net.coef_})
selected_features = features[features['Coefficient'] != 0]

# Extract the names of the selected features
selected_feature_names = selected_features['Feature'].values

# Combine the selected features with the original data's first six columns
combined_data = pd.concat([data.iloc[:, :6], data[selected_feature_names]], axis=1)

# Save the combined data to a new CSV file
combined_data.to_csv('C:/Users/theoa/OneDrive/Desktop/Bath/Research Project 1B/test/selected_features_data.csv', index=False)
